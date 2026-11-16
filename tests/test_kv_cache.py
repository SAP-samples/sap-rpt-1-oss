# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0
"""KV-cache parity tests.

The cache is exact by construction (context rows never attend to query rows), so cached and uncached
predictions must agree up to floating-point reduction order. The tests use randomly initialised weights,
so they need neither the gated checkpoint nor a GPU (the GPU case is skipped without one).
"""
import numpy as np
import pandas as pd
import pytest
import torch

import sap_rpt_oss.rpt as rpt_module
from sap_rpt_oss import SAP_RPT_OSS_Classifier, SAP_RPT_OSS_Regressor
from sap_rpt_oss.model.torch_model import RPT


@pytest.fixture(autouse=True)
def random_weights(monkeypatch):
    """Skip the checkpoint download; keep the random initialisation (seeded)."""
    monkeypatch.setattr(rpt_module, 'hf_hub_download', lambda **kwargs: 'unused')
    monkeypatch.setattr(RPT, 'load_weights', lambda self, *args, **kwargs: None)
    torch.manual_seed(0)


@pytest.fixture
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)


def make_table(n_rows=420, seed=0):
    """Numeric columns with missing values, a string column and a date column; no constant columns."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n_rows, 8)), columns=[f'feature_{i}' for i in range(8)])
    X = X.mask(rng.random(X.shape) < 0.1)
    X['colour'] = rng.choice(['red', 'green', 'blue', 'black'], n_rows)
    X['opened'] = pd.to_datetime('2020-01-01') + pd.to_timedelta(rng.integers(0, 2000, n_rows), unit='D')
    logit = X['feature_0'].fillna(0) - 0.5 * X['feature_1'].fillna(0) + (X['colour'] == 'red')
    y_class = (logit + rng.normal(size=n_rows) > 0.3).astype(int)
    y_reg = logit + 0.1 * rng.normal(size=n_rows)
    return X, y_class, y_reg


def fit_pair(estimator_cls, X, y, **kwargs):
    plain = estimator_cls(kv_cache=False, **kwargs).fit(X, y)
    torch.manual_seed(0)
    cached = estimator_cls(kv_cache=True, **kwargs).fit(X, y)
    cached.model.load_state_dict(plain.model.state_dict())
    return plain, cached


SETTINGS = dict(max_context_size=256, bagging=3, test_chunk_size=64)


def test_classifier_parity_cpu(cpu_only):
    X, y, _ = make_table()
    plain, cached = fit_pair(SAP_RPT_OSS_Classifier, X.iloc[:300], y.iloc[:300], **SETTINGS)
    p_plain, p_cached = plain.predict_proba(X.iloc[300:]), cached.predict_proba(X.iloc[300:])
    assert np.isfinite(p_cached).all()
    np.testing.assert_allclose(p_cached, p_plain, atol=1e-5)
    assert cached.predict(X.iloc[300:]) == plain.predict(X.iloc[300:])


def test_regressor_parity_cpu(cpu_only):
    X, _, y = make_table()
    plain, cached = fit_pair(SAP_RPT_OSS_Regressor, X.iloc[:300], y.iloc[:300], **SETTINGS)
    np.testing.assert_allclose(cached.predict(X.iloc[300:]), plain.predict(X.iloc[300:]), atol=1e-4)


def test_cached_predictions_do_not_depend_on_chunking(cpu_only):
    X, y, _ = make_table()
    model = SAP_RPT_OSS_Classifier(kv_cache=True, **SETTINGS).fit(X.iloc[:300], y.iloc[:300])
    reference = model.predict_proba(X.iloc[300:])
    for chunk in (7, 1000):
        model.test_chunk_size = chunk
        np.testing.assert_allclose(model.predict_proba(X.iloc[300:]), reference, atol=1e-6)


def test_single_row_chunks(cpu_only):
    X, y_class, y_reg = make_table()
    clf = SAP_RPT_OSS_Classifier(kv_cache=True, **SETTINGS).fit(X.iloc[:300], y_class.iloc[:300])
    probs = clf.predict_proba(X.iloc[300:365])                # 65 rows with chunks of 64 -> one trailing row
    assert probs.shape == (65, 2) and np.isfinite(probs).all()
    reg = SAP_RPT_OSS_Regressor(kv_cache=True, **SETTINGS).fit(X.iloc[:300], y_reg.iloc[:300])
    assert reg.predict(X.iloc[300:301]).shape == (1,)
    assert reg.predict(X.iloc[300:365]).shape == (65,)


def count_cache_builds(monkeypatch):
    calls = []
    original = RPT.build_context_cache

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RPT, 'build_context_cache', counting)
    return calls


def test_cache_is_reused_across_calls_and_reset_by_fit(cpu_only, monkeypatch):
    X, y, _ = make_table()
    builds = count_cache_builds(monkeypatch)
    model = SAP_RPT_OSS_Classifier(kv_cache=True, **SETTINGS).fit(X.iloc[:300], y.iloc[:300])
    first = model.predict_proba(X.iloc[300:360])
    assert len(builds) == model.bagging_number == len(model._kv_caches)
    second = model.predict_proba(X.iloc[360:])                # new rows, same contexts: no rebuild
    assert len(builds) == model.bagging_number
    np.testing.assert_allclose(np.concatenate([first, second]), model.predict_proba(X.iloc[300:]), atol=1e-6)
    model.fit(X.iloc[:300], y.iloc[:300])                     # refit drops the caches
    assert model._kv_caches == {}
    model.clear_kv_cache()
    model.predict_proba(X.iloc[300:310])
    assert len(builds) == 2 * model.bagging_number


def test_per_call_mode_matches_and_keeps_nothing(cpu_only, monkeypatch):
    X, y, _ = make_table()
    kept = SAP_RPT_OSS_Classifier(kv_cache=True, **SETTINGS).fit(X.iloc[:300], y.iloc[:300])
    torch.manual_seed(0)
    per_call = SAP_RPT_OSS_Classifier(kv_cache='per_call', **SETTINGS).fit(X.iloc[:300], y.iloc[:300])
    per_call.model.load_state_dict(kept.model.state_dict())
    np.testing.assert_allclose(per_call.predict_proba(X.iloc[300:]), kept.predict_proba(X.iloc[300:]), atol=1e-6)
    assert per_call._kv_caches == {}


def test_pickle_drops_the_cache(cpu_only):
    import pickle
    X, y, _ = make_table()
    model = SAP_RPT_OSS_Classifier(kv_cache=True, **SETTINGS).fit(X.iloc[:300], y.iloc[:300])
    reference = model.predict_proba(X.iloc[300:])
    assert len(model._kv_caches) == model.bagging_number
    restored = pickle.loads(pickle.dumps(model))
    assert restored._kv_caches == {}
    np.testing.assert_allclose(restored.predict_proba(X.iloc[300:]), reference, atol=1e-6)


@pytest.mark.parametrize('value', ['always', 1, None])
def test_invalid_kv_cache_value(value):
    with pytest.raises(ValueError):
        SAP_RPT_OSS_Classifier(kv_cache=value)
    model = SAP_RPT_OSS_Classifier()
    model.set_params(kv_cache=value)                          # set_params skips __init__: checked at predict time
    X, y, _ = make_table(80)
    with pytest.raises(ValueError):
        model.fit(X.iloc[:60], y.iloc[:60]).predict_proba(X.iloc[60:])


def test_numpy_input_with_unnamed_target(cpu_only):
    # pd.DataFrame(X) names the features 0..n-1 and y.to_frame() names the target 0: the names collide
    X, y, _ = make_table()
    X = X.select_dtypes('number').to_numpy()
    plain, cached = fit_pair(SAP_RPT_OSS_Classifier, X[:300], pd.Series(y.to_numpy()[:300]), **SETTINGS)
    np.testing.assert_allclose(cached.predict_proba(X[300:]), plain.predict_proba(X[300:]), atol=1e-5)


def test_test_columns_are_matched_by_name(cpu_only):
    X, y, _ = make_table()
    plain, cached = fit_pair(SAP_RPT_OSS_Classifier, X.iloc[:300], y.iloc[:300], **SETTINGS)
    shuffled = X.iloc[300:, ::-1]
    np.testing.assert_allclose(cached.predict_proba(shuffled), plain.predict_proba(shuffled), atol=1e-5)


def test_parameter_change_invalidates_the_cache(cpu_only, monkeypatch):
    X, y, _ = make_table()
    builds = count_cache_builds(monkeypatch)
    model = SAP_RPT_OSS_Classifier(kv_cache=True, **SETTINGS).fit(X.iloc[:300], y.iloc[:300])
    model.predict_proba(X.iloc[300:])
    model.set_params(max_context_size=128)
    probs = model.predict_proba(X.iloc[300:])
    assert len(builds) == 2 * model.bagging_number
    assert all(len(entry['X']) == 128 for entry in model._kv_caches.values())
    reference = SAP_RPT_OSS_Classifier(kv_cache=False, **dict(SETTINGS, max_context_size=128))
    reference.model.load_state_dict(model.model.state_dict())
    np.testing.assert_allclose(probs, reference.fit(X.iloc[:300], y.iloc[:300]).predict_proba(X.iloc[300:]), atol=1e-5)


def test_clear_kv_cache_does_not_mark_the_model_fitted():
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted
    model = SAP_RPT_OSS_Classifier(kv_cache=True)
    model.clear_kv_cache()
    with pytest.raises(NotFittedError):
        check_is_fitted(model)


def test_query_dtypes_are_aligned_with_the_context(cpu_only):
    X, y, _ = make_table()
    model = SAP_RPT_OSS_Classifier(kv_cache=True, **SETTINGS).fit(X.iloc[:300], y.iloc[:300])
    reference = model.predict_proba(X.iloc[300:])
    as_text = X.iloc[300:].astype({'feature_2': object})     # numbers that arrive as objects are converted
    np.testing.assert_allclose(model.predict_proba(as_text), reference, atol=1e-6)
    broken = as_text.copy()
    broken.iloc[0, broken.columns.get_loc('feature_2')] = 'n/a'
    with pytest.raises(ValueError, match='feature_2'):
        model.predict_proba(broken)


def test_unpickling_an_estimator_saved_before_the_cache_existed(cpu_only):
    import pickle
    X, y, _ = make_table()
    model = SAP_RPT_OSS_Classifier(**SETTINGS).fit(X.iloc[:300], y.iloc[:300])
    reference = model.predict_proba(X.iloc[300:])
    state = model.__getstate__()
    for name in ('kv_cache', 'cache_device', 'max_cells_per_batch', 'verbose', '_kv_caches', '_kv_cache_signature'):
        state.pop(name)
    old = SAP_RPT_OSS_Classifier.__new__(SAP_RPT_OSS_Classifier)
    old.__setstate__(state)
    restored = pickle.loads(pickle.dumps(old))
    assert restored.get_params()['kv_cache'] is False
    np.testing.assert_allclose(restored.predict_proba(X.iloc[300:]), reference, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_host_cache_is_page_locked_and_released():
    X, y, _ = make_table()
    model = SAP_RPT_OSS_Classifier(kv_cache=True, cache_device='cpu', **SETTINGS).fit(X.iloc[:300], y.iloc[:300])
    model.predict_proba(X.iloc[300:])
    caches = [entry['cache'] for entry in model._kv_caches.values()]
    keys, _ = caches[0][0]
    assert keys.device.type == 'cpu' and keys.is_pinned()
    model.clear_kv_cache()
    assert all(len(cache) == 0 and not cache._registered for cache in caches)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
@pytest.mark.parametrize('cache_device', ['cuda', 'cpu'])
def test_classifier_parity_gpu(cache_device):
    X, y, _ = make_table()
    plain, cached = fit_pair(SAP_RPT_OSS_Classifier, X.iloc[:300], y.iloc[:300], cache_device=cache_device,
                             **SETTINGS)
    p_plain, p_cached = plain.predict_proba(X.iloc[300:]), cached.predict_proba(X.iloc[300:])
    assert np.isfinite(p_cached).all()
    np.testing.assert_allclose(p_cached, p_plain, atol=2e-3)

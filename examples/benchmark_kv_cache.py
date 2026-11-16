# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0
"""Speed, peak GPU memory and agreement of kv_cache vs the uncached path, on synthetic data.

Two scenarios:
  batch     one predict_proba call on many test rows
  repeated  many predict_proba calls on a few rows each (e.g. online scoring, perturbation-based explanations)

    python examples/benchmark_kv_cache.py --out kv_cache_benchmark.md
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from sap_rpt_oss import SAP_RPT_OSS_Classifier


def make_table(n_rows, n_columns, seed=0):
    rng = np.random.default_rng(seed)
    n_text = max(1, n_columns // 10)
    X = pd.DataFrame(rng.normal(size=(n_rows, n_columns - n_text)),
                     columns=[f'measurement_{i}' for i in range(n_columns - n_text)])
    X = X.mask(rng.random(X.shape) < 0.05)
    for j in range(n_text):
        X[f'category_{j}'] = rng.choice(['north', 'south', 'east', 'west', 'centre'], n_rows)
    logit = X.iloc[:, 0].fillna(0) - X.iloc[:, 1].fillna(0) + (X['category_0'] == 'north')
    y = (logit + rng.normal(size=n_rows) > 0.5).astype(int)
    return X, y


def timed_predict(model, X_test, repeats=1, rows_per_call=None):
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    if rows_per_call is None:
        probs = model.predict_proba(X_test)
    else:
        probs = np.concatenate([model.predict_proba(X_test.iloc[i * rows_per_call:(i + 1) * rows_per_call])
                                for i in range(repeats)])
    torch.cuda.synchronize()
    return probs, time.perf_counter() - start, torch.cuda.max_memory_allocated() / 2**30


def run(label, factory, X_train, y_train, X_test, **kwargs):
    try:
        model = factory().fit(X_train, y_train)
        probs, seconds, peak = timed_predict(model, X_test, **kwargs)
        del model
        return dict(label=label, probs=probs, seconds=seconds, peak=peak)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(label=label, probs=None, seconds=None, peak=None)


def cell(result, baseline=None, memory=True):
    """'seconds / peak GiB / speedup' (or 'OOM') for a markdown table."""
    if result['probs'] is None:
        return 'OOM'
    parts = [f"{result['seconds']:.1f} s"]
    if memory:
        parts.append(f"{result['peak']:.1f} GiB")
    if baseline is not None and baseline['probs'] is not None:
        parts.append(f"{baseline['seconds'] / result['seconds']:.1f}x")
    return ' / '.join(parts)


def max_diff(results, baseline):
    diffs = [np.max(np.abs(r['probs'] - baseline['probs'])) for r in results
             if r['probs'] is not None and baseline['probs'] is not None]
    return f'{max(diffs):.1e}' if diffs else 'n/a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--contexts', default='2048,4096,8192')
    ap.add_argument('--columns', default='20,100')
    ap.add_argument('--n_test', type=int, default=8192)
    ap.add_argument('--repeated_context', type=int, default=2048)
    ap.add_argument('--repeated_calls', type=int, default=50)
    ap.add_argument('--rows_per_call', type=int, default=16)
    ap.add_argument('--small_gpu_cells', type=int, default=131072)
    ap.add_argument('--sections', default='batch,repeated')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    dtype = 'bf16' if torch.cuda.get_device_capability(0)[0] >= 8 else 'fp16'
    sections = a.sections.split(',')
    lines = [f'GPU: {torch.cuda.get_device_name(0)} ({dtype}), bagging 1, synthetic data.']
    if 'batch' in sections:
        lines += ['', '### One predict call', '',
                  f'{a.n_test:,} test rows; uncached with test_chunk_size=1000, cached with 2048. '
                  'Each cell: seconds / peak GPU memory / speedup over uncached.', '',
                  '| columns | context | uncached | kv_cache=True | '
                  f'kv_cache=True, max_cells_per_batch={a.small_gpu_cells:,} | max abs dP |',
                  '|---:|---:|---|---|---|---:|']
        for n_columns in [int(c) for c in a.columns.split(',')]:
            for context in [int(c) for c in a.contexts.split(',')]:
                X, y = make_table(context + a.n_test, n_columns)
                X_train, y_train, X_test = X.iloc[:context], y.iloc[:context], X.iloc[context:]
                common = dict(max_context_size=context, bagging=1)
                stock = run('uncached', lambda: SAP_RPT_OSS_Classifier(test_chunk_size=1000, **common),
                            X_train, y_train, X_test)
                cached = run('kv_cache', lambda: SAP_RPT_OSS_Classifier(kv_cache=True, test_chunk_size=2048, **common),
                             X_train, y_train, X_test)
                small = run('kv_cache small', lambda: SAP_RPT_OSS_Classifier(
                    kv_cache=True, test_chunk_size=2048, max_cells_per_batch=a.small_gpu_cells, **common),
                    X_train, y_train, X_test)
                lines.append(f'| {n_columns} | {context:,} | {cell(stock)} | {cell(cached, stock)} | '
                             f'{cell(small, stock)} | {max_diff([cached, small], stock)} |')
                print(lines[-1], flush=True)

    if 'repeated' in sections:
        context = a.repeated_context
        lines += ['', '### Repeated small calls', '',
                  f'{a.repeated_calls} predict_proba calls of {a.rows_per_call} rows each, context {context:,}. '
                  'Each cell: total seconds / speedup over uncached.', '',
                  "| columns | uncached | kv_cache=True | kv_cache=True, cache_device='cuda' | max abs dP |",
                  '|---:|---|---|---|---:|']
        for n_columns in [int(c) for c in a.columns.split(',')]:
            X, y = make_table(context + a.repeated_calls * a.rows_per_call, n_columns, seed=1)
            X_train, y_train, X_test = X.iloc[:context], y.iloc[:context], X.iloc[context:]
            calls = dict(repeats=a.repeated_calls, rows_per_call=a.rows_per_call)
            common = dict(max_context_size=context, bagging=1)
            stock = run('uncached', lambda: SAP_RPT_OSS_Classifier(**common), X_train, y_train, X_test, **calls)
            cached = run('kv_cache', lambda: SAP_RPT_OSS_Classifier(kv_cache=True, **common),
                         X_train, y_train, X_test, **calls)
            on_gpu = run('kv_cache cuda', lambda: SAP_RPT_OSS_Classifier(kv_cache=True, cache_device='cuda', **common),
                         X_train, y_train, X_test, **calls)
            lines.append(f'| {n_columns} | {cell(stock, memory=False)} | {cell(cached, stock, memory=False)} | '
                         f'{cell(on_gpu, stock, memory=False)} | {max_diff([cached, on_gpu], stock)} |')
            print(lines[-1], flush=True)
    text = '\n'.join(lines)
    print(text)
    if a.out:
        with open(a.out, 'w') as f:
            f.write(text + '\n')


if __name__ == '__main__':
    main()

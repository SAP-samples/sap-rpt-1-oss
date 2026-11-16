# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

import warnings
from abc import ABC, abstractmethod
from math import ceil
from pathlib import Path
from typing import Literal, Optional, Union

from huggingface_hub import hf_hub_download
import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted

from sap_rpt_oss.constants import ModelSize
from sap_rpt_oss.data.tokenizer import Tokenizer
from sap_rpt_oss.model.attention import TwoDimensionalAttentionLayer
from sap_rpt_oss.model.torch_model import RPT

warnings.filterwarnings('ignore', message='.*not support non-writable tensors.*')


def to_device(x, device: Union[torch.device, int], dtype: Optional[torch.dtype] = None, raise_on_unexpected=True):
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            target_dtype = dtype if v.dtype == torch.float32 else v.dtype
            x[k] = v.to(device, dtype=target_dtype)
        elif isinstance(v, dict):
            x[k] = to_device(v, device, dtype=dtype)
        elif v is not None and raise_on_unexpected:
            raise ValueError(f'Unknown type, {type(v)}')
    return x


def slice_rows(data: dict, rows: slice) -> dict:
    """Row slice of a tokenized data dict ('column_embeddings' is shared by all rows)."""
    return {k: (v if k == 'column_embeddings' else v[rows]) for k, v in data.items()}


class SAP_RPT_OSS_Estimator(BaseEstimator, ABC):
    """SAP_RPT_OSS_Estimator (sap-rpt-1-oss) class.

    Args:
        checkpoint: path to the checkpoint file; must be of size base model size
        bagging: number of bagging iterations; if 1 there is no bagging. If 'auto', then:
            - There is no bagging if the number of samples is less than max_context_size - just use everything
            - Otherwise, the training data is split into chunks size max_context_size rows:
            ceil(len(dataset) / max_context_size) overlapping chunks and each chunk is then used as a bagging
            iteration (capped at MAX_AUTO_BAGS = 16).
        max_context_size: maximum number of samples to use for training
        num_regression_bins: number of bins to use for regression (to convert into classification).
            Unused if regression_type is 'l2'.
        regression_type: regression type that was used in the specified model
            - reg-as-classif - binned regression where bin is associated with the quantile of a given column
            - l2 - direct prediction of the target value with L2 loss during training
        classification_type: classification type that was used in the specified model
            - cross-entropy - class likelihood prediction using cross entropy loss during training
            - clustering - class prediction using similarity between context and query vectors
            - clustering-cosine - class prediction using cosine similarity between context and query vectors 
        drop_constant_columns: flag to indicate to drop constant columns in the input dataframe
        test_chunk_size: Batch size of test rows to use for prediction at once
        kv_cache: cache the context of each bagging iteration. Context rows never attend to test rows, so their
            cross-row keys/values can be computed once and reused: each test chunk then runs only its own rows
            through the encoder. Predictions match kv_cache=False up to floating-point rounding, except for the
            constant-column rule below.
            - False: no cache (default).
            - True: each bagging iteration's cache is built on first use after fit and kept until the next fit,
              a change of the parameters that define the contexts, or clear_kv_cache() (needed after changing the
              model weights), so repeated predict calls reuse it. Memory per bagging iteration:
              2 * num_layers * num_columns * context_rows * hidden_size * bytes per element
              (12 * 2 * 768 * 2 bytes = 36 KiB per context cell in fp16/bf16).
            - 'per_call': same computation, but each cache is freed once its bagging iteration has scored the
              rows of the current predict call (one cache in memory at a time).
            With a cache, constant columns are detected on the context rows only (without it, on context + test
            chunk), so predictions no longer depend on how the test rows are chunked.
        cache_device: where the KV cache lives: 'cuda', 'cpu' (page-locked host memory, streamed to the GPU),
            or 'auto' (GPU if the cache takes less than half of the free GPU memory, else CPU).
        max_cells_per_batch: KV-cache path only: maximum number of cells (rows x columns) processed at once by
            each attention sub-layer. None keeps the default (8192 x 100, the budget the uncached path uses).
            Lower it to fit larger contexts on smaller GPUs, e.g. 131072 runs an 8192-row, 135-column context
            in about 12 GB.
        verbose: show a progress bar over test chunks on the KV-cache path.
    """
    classification_or_regression: str
    MAX_AUTO_BAGS = 16
    MAX_NUM_COLUMNS = 500

    def __init__(self,
                 checkpoint: str = '2025-11-04_sap-rpt-one-oss.pt',
                 bagging: Union[Literal['auto'], int] = 8,
                 max_context_size: int = 8192,
                 drop_constant_columns: bool = True,
                 test_chunk_size: int = 1000,
                 kv_cache: Union[bool, Literal['per_call']] = False,
                 cache_device: str = 'auto',
                 max_cells_per_batch: Optional[int] = None,
                 verbose: bool = False):

        self.model_size = ModelSize.base
        self.checkpoint = checkpoint
        self.regression_type = "l2"
        self.classification_type = "cross-entropy"
        self.test_chunk_size = test_chunk_size
        self.kv_cache = kv_cache
        self.cache_device = cache_device
        self.max_cells_per_batch = max_cells_per_batch
        self.verbose = verbose
        self._kv_cache_mode()  # validates kv_cache
        if cache_device not in ('auto', 'cuda', 'cpu'):
            raise ValueError("cache_device must be 'auto', 'cuda' or 'cpu'")
        self._checkpoint_path = hf_hub_download(repo_id="SAP/sap-rpt-1-oss", filename=checkpoint)
        self.bagging = bagging
        if not isinstance(bagging, int) and bagging != 'auto':
            raise ValueError('bagging must be an integer or "auto"')
        self.max_context_size = max_context_size
        self.num_regression_bins = 16
        self.model = RPT(self.model_size, regression_type=self.regression_type, classification_type=self.classification_type)
        # We're using a single GPU here, even if more are available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            if torch.cuda.get_device_capability(0)[0] >= 8:
                self.dtype = torch.bfloat16
            else:
                self.dtype = torch.float16
            self.model = self.model.to(dtype=self.dtype)
        else:
            self.dtype = torch.float32

        self.model.load_weights(Path(self._checkpoint_path), self.device)
        self.seed = 42
        self.drop_constant_columns = drop_constant_columns
        self.tokenizer = Tokenizer(
            regression_type=self.regression_type,
            classification_type=self.classification_type,
            random_seed=self.seed,
            num_regression_bins=self.num_regression_bins,
            is_valid=True)
        self.model.to(self.device).eval()

    @abstractmethod
    def task_specific_fit(self):
        pass

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Union[pd.Series, np.ndarray]):
        """Fit the model.

        Args:
            X: The input dataframe.
            y: The target column.
        """
        if len(X) != len(y):
            raise ValueError('X and y must have the same length')
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if not isinstance(y, pd.Series):
            y = pd.Series(y, name='TARGET')

        self.clear_kv_cache()
        self.X_ = X

        self.bagging_config = self.bagging
        if X.shape[0] < self.max_context_size:
            self.bagging_config = 1

        self.y_ = y
        # Return the classifier
        self.task_specific_fit()
        return self

    @property
    def bagging_number(self):
        check_is_fitted(self)
        if self.bagging_config == 'auto':
            return min(self.MAX_AUTO_BAGS, ceil(len(self.X_) / self.max_context_size))
        else:
            assert isinstance(self.bagging_config, int)
            return self.bagging_config

    def bag_training_frame(self, bagging_index) -> pd.DataFrame:
        """Context rows (features + target column) used by one bagging iteration."""
        df_train = pd.concat([self.X_, self.y_.to_frame()], axis=1)
        if isinstance(self.bagging_config, int) and self.bagging_config > 1:
            # For bagging, we use replacement
            df_train = df_train.sample(self.max_context_size, replace=False, random_state=self.seed + bagging_index)
        elif len(df_train) > self.max_context_size:
            if isinstance(self.bagging_config, str):
                assert self.bagging_config == 'auto'
                # Split the data into overlapping chunks of size max_context_size
                # Randomize order as well, to have balanced bags
                # bagging_index = 0 --> select 0:max_context_size
                # ... (linearly spaced like np.linspace(0, len(df_train) - max_context_size, self.bagging_number))
                # bagging_index = self.bagging_number - 1 --> select (len(df_train) - max_context_size):len(df_train)
                start = int((len(df_train) - self.max_context_size) / (self.bagging_number - 1) * bagging_index)
                # We need a fixed seed, so across diffent "bagging folds" we select the correct indices
                # (as non-overlapping as possible)
                np.random.seed(self.seed)
                indices = np.random.permutation(df_train.index)
                end = start + self.max_context_size
                df_train = df_train.loc[indices[start:end]]
            else:
                # There is no bagging, but we still have to sample because there are too many points
                df_train = df_train.sample(self.max_context_size, replace=False, random_state=self.seed + bagging_index)
        return df_train

    def get_tokenized_data(self, X_test, bagging_index):
        X_train = self.X_
        y_train = self.y_

        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test, columns=X_train.columns)
        y_test = pd.Series([y_train.iloc[0]] * len(X_test), name=self.y_.name, index=X_test.index)

        df_train = self.bag_training_frame(bagging_index)
        df_test = pd.concat([X_test, y_test.to_frame()], axis=1)

        df = pd.concat([df_train, df_test], ignore_index=True)

        if self.drop_constant_columns:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            constant_cols = list(X.columns[X.nunique() == 1])
            if constant_cols:
                X = X.drop(columns=constant_cols)
                df = pd.concat([X, y], axis=1)

        if df.shape[1] > self.MAX_NUM_COLUMNS:
            X = df.iloc[:, :-1]
            y = df.iloc[:, -1:]
            X = X.sample(n=self.MAX_NUM_COLUMNS - 1, axis=1, random_state=self.seed + bagging_index, replace=False)
            df = pd.concat([X, y], axis=1)

        df_train = df.iloc[:len(df_train)]
        df_test = df.iloc[len(df_train):]
        X_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1:]
        X_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1:]

        data, labels, label_classes = self.tokenizer(X_train, y_train, X_test, y_test,
                                                     self.classification_or_regression)

        target_mean, target_std = 0, 0
        is_regression = self.classification_or_regression == 'regression'
        if is_regression and self.regression_type == 'l2':
            _, target_mean, target_std = self.tokenizer.standard_scale_column(y_train, y_test)

        return {
            'data': data,
            'num_rows': df.shape[0],
            'num_cols': df.shape[1],
            'labels': None,
            'is_regression': torch.tensor(is_regression),
            'label_classes': np.asarray(label_classes),
            'target_mean': target_mean,
            'target_std': target_std
        }

    def _kv_cache_mode(self) -> Optional[str]:
        """None (no cache), 'keep' (kv_cache=True) or 'per_call'; raises on any other value."""
        mode = getattr(self, 'kv_cache', False)
        if isinstance(mode, (bool, np.bool_)):
            return 'keep' if mode else None
        if isinstance(mode, str) and mode == 'per_call':
            return 'per_call'
        raise ValueError(f"kv_cache must be False, True or 'per_call', got {mode!r}")

    def bag_columns(self, df_train: pd.DataFrame, bagging_index) -> list:
        """
        Positions of the feature columns used by one bagging iteration (KV-cache path), decided from its context
        rows only: constant columns are dropped, then at most MAX_NUM_COLUMNS - 1 are sampled as in
        get_tokenized_data.
        """
        X = df_train.iloc[:, :-1]
        positions = np.arange(X.shape[1])
        if self.drop_constant_columns:
            positions = positions[X.nunique().to_numpy() != 1]
        if len(positions) + 1 > self.MAX_NUM_COLUMNS:
            by_position = X.iloc[:, positions].set_axis(range(len(positions)), axis=1)
            picked = by_position.sample(n=self.MAX_NUM_COLUMNS - 1, axis=1, random_state=self.seed + bagging_index,
                                        replace=False).columns
            positions = positions[np.asarray(picked)]
        return list(positions)

    def resolve_cache_device(self, num_context: int, num_columns: int) -> torch.device:
        if self.device.type != 'cuda':
            return self.device
        if self.cache_device != 'auto':
            return torch.device(self.cache_device)
        config = self.model.config
        item_size = torch.empty((), dtype=self.dtype).element_size()
        needed = 2 * config.num_hidden_layers * num_columns * num_context * config.hidden_size * item_size
        free, _ = torch.cuda.mem_get_info(self.device)
        # memory held by PyTorch's caching allocator but not in use is available to us as well
        free += torch.cuda.memory_reserved(self.device) - torch.cuda.memory_allocated(self.device)
        return self.device if needed < 0.5 * free else torch.device('cpu')

    def clear_kv_cache(self):
        """Free the KV caches kept by kv_cache=True; they are rebuilt on the next predict call. Call it after
        changing the model weights (fit and parameter changes that affect the contexts already do this)."""
        for entry in getattr(self, '_kv_caches', {}).values():
            if entry.get('cache') is not None:
                entry['cache'].release()
        self._kv_caches = {}
        self._kv_cache_signature = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __getstate__(self):
        # never pickle the KV caches (they can be tens of GB); they are rebuilt on demand
        state = self.__dict__.copy()
        state['_kv_caches'] = {}
        state['_kv_cache_signature'] = None
        return state

    def __setstate__(self, state):
        # estimators pickled before the KV cache existed
        for name, default in (('kv_cache', False), ('cache_device', 'auto'), ('max_cells_per_batch', None),
                              ('verbose', False), ('_kv_caches', {}), ('_kv_cache_signature', None)):
            state.setdefault(name, default)
        self.__dict__.update(state)

    def _context_signature(self) -> tuple:
        """Everything (besides the weights) that the cached contexts depend on."""
        return (self.bagging_config, self.max_context_size, self.seed, self.drop_constant_columns,
                self.MAX_NUM_COLUMNS)

    def _context_entry(self, bagging_index) -> dict:
        """Context rows, columns and (once built) the KV cache of one bagging iteration."""
        entry = self._kv_caches.get(bagging_index)
        if entry is None:
            df_train = self.bag_training_frame(bagging_index)
            positions = self.bag_columns(df_train, bagging_index)
            entry = {'positions': positions, 'X': df_train.iloc[:, positions], 'y': df_train.iloc[:, -1:],
                     'cache': None}
        return entry

    @staticmethod
    def _align_query_dtypes(X_context: pd.DataFrame, X_query: pd.DataFrame) -> pd.DataFrame:
        """
        The uncached path tokenizes context and test rows as one frame, so pandas reconciles their dtypes. Here they
        are tokenized separately: test columns that are numeric in the context are converted to numbers, and a
        column that cannot be converted raises instead of being embedded differently from the context.
        """
        converted = {}
        for position in range(X_context.shape[1]):
            context_dtype, query_column = X_context.dtypes.iloc[position], X_query.iloc[:, position]
            if (pd.api.types.is_numeric_dtype(context_dtype) and not pd.api.types.is_bool_dtype(context_dtype)
                    and not pd.api.types.is_numeric_dtype(query_column.dtype)):
                try:
                    converted[position] = pd.to_numeric(query_column, errors='raise')
                except (ValueError, TypeError) as error:
                    raise ValueError(f'kv_cache: column {X_context.columns[position]!r} is numeric in the context '
                                     'rows but has non-numeric values in the test rows; convert it or use '
                                     'kv_cache=False') from error
        if converted:
            X_query = X_query.copy()
            for position, column in converted.items():
                X_query.isetitem(position, column)
        return X_query

    @torch.no_grad()
    def cached_bag_outputs(self, X: pd.DataFrame):
        """
        KV-cache path. For each bagging iteration, builds (or reuses) the context cache and runs every test
        chunk against it. Yields one list per bagging iteration of (outputs, targets, label_classes, target_mean,
        target_std) per test chunk, in the order of X.
        """
        if self.classification_type in ['clustering', 'clustering-cosine']:
            raise NotImplementedError('kv_cache supports classification_type="cross-entropy" only')
        keep = self._kv_cache_mode() == 'keep'
        if X.columns.is_unique and self.X_.columns.is_unique and not X.columns.equals(self.X_.columns):
            X = X[self.X_.columns]  # match test columns to the context by name, as pd.concat does on the uncached path
        if self._kv_cache_signature != self._context_signature():
            self.clear_kv_cache()
            self._kv_cache_signature = self._context_signature()
        is_regression = self.classification_or_regression == 'regression'
        for layer in self.model.in_context_encoder:
            layer.max_cells_per_batch = self.max_cells_per_batch or TwoDimensionalAttentionLayer.max_cells_per_batch
        for bagging_index in range(self.bagging_number):
            entry = self._context_entry(bagging_index)
            X_context, y_context = entry['X'], entry['y']
            num_context = len(X_context)
            target_mean, target_std = 0, 0
            if is_regression and self.regression_type == 'l2':
                _, target_mean, target_std = self.tokenizer.standard_scale_column(y_context, y_context)
            outputs = []
            step = self.test_chunk_size
            bounds = [(start, min(start + step, len(X))) for start in range(0, len(X), step)]
            if self.verbose:
                from tqdm.auto import tqdm
                bounds = tqdm(bounds, desc=f'bag {bagging_index} test chunks', leave=False)
            for start, end in bounds:
                X_query = self._align_query_dtypes(X_context, X.iloc[start:end, entry['positions']])
                y_query = pd.Series([self.y_.iloc[0]] * len(X_query), name=self.y_.name,
                                    index=X_query.index).to_frame()
                data, _, label_classes = self.tokenizer(X_context, y_context, X_query, y_query,
                                                        self.classification_or_regression)
                if entry['cache'] is None:
                    context_data = to_device(slice_rows(data, slice(0, num_context)), self.device,
                                             raise_on_unexpected=False, dtype=self.dtype)
                    cache_device = self.resolve_cache_device(num_context, data['column_embeddings'].shape[0])
                    entry['cache'] = self.model.build_context_cache(context_data, is_regression, cache_device)
                    del context_data
                query_data = to_device(slice_rows(data, slice(num_context, None)), self.device,
                                       raise_on_unexpected=False, dtype=self.dtype)
                out = self.model.forward_with_cache(query_data, is_regression, entry['cache'])
                outputs.append((out, query_data['target'], np.asarray(label_classes), target_mean, target_std))
            if keep:
                self._kv_caches[bagging_index] = entry
            elif entry['cache'] is not None:
                entry['cache'].release()
            del entry
            yield outputs

    @abstractmethod
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[list, np.ndarray]:
        pass


class SAP_RPT_OSS_Classifier(ClassifierMixin, SAP_RPT_OSS_Estimator):
    classification_or_regression = 'classification'

    def task_specific_fit(self):
        # Store the classes seen during fit
        self.classes_ = unique_labels(self.y_)

    def reorder_logits(self, logits, tokenized_classes):
        class_mapping = {cls: idx for idx, cls in enumerate(self.classes_)}
        indices = np.array([class_mapping[cls] for cls in tokenized_classes])
        new_logits = torch.full((logits.shape[0], len(self.classes_)),
                                float('-inf'),
                                dtype=logits.dtype,
                                device=logits.device)
        new_logits[:, indices] = logits[:, :len(tokenized_classes)]
        return new_logits

    @torch.no_grad()
    def _predict(self, X: Union[pd.DataFrame, np.ndarray]):
        # Check if fit has been called
        check_is_fitted(self)

        all_logits = []

        for bagging_index in range(self.bagging_number):
            tokenized_data = self.get_tokenized_data(X.copy(), bagging_index)

            try:
                tokenized_data = to_device(tokenized_data, self.device, raise_on_unexpected=False, dtype=self.dtype)
            except TypeError:
                # Legacy compatibility
                tokenized_data = to_device(tokenized_data, self.device, raise_on_unexpected=False)
            logits_classif = self.model(**tokenized_data)

            _, logits = self.model.extract_prediction_classification(logits_classif, tokenized_data['data']['target'],
                                                                     tokenized_data['label_classes'])

            all_logits.append(self.reorder_logits(logits, tokenized_data['label_classes']))

        all_logits = torch.stack(all_logits)
        if self.classification_type in ['clustering', 'clustering-cosine']:
            # Pick the class of the most similar element across folds, then softmax
            all_logits = torch.max(all_logits, dim=0).values.cpu().float()
            probs = torch.nn.functional.softmax(all_logits, dim=-1)
        else:
            # Average probabilities across folds, i.e. first softmax, then mean
            all_probs = torch.nn.functional.softmax(all_logits, dim=-1)
            probs = all_probs.mean(dim=0)

        return probs

    @torch.no_grad()
    def _predict_proba_cached(self, X: pd.DataFrame) -> torch.Tensor:
        check_is_fitted(self)
        bag_probs = []
        for outputs in self.cached_bag_outputs(X):
            chunks = []
            for out, targets, label_classes, _, _ in outputs:
                _, logits = self.model.extract_prediction_classification(out, targets, label_classes)
                chunks.append(torch.nn.functional.softmax(self.reorder_logits(logits, label_classes), dim=-1))
            bag_probs.append(torch.cat(chunks))
        # same as the uncached path: softmax per bagging iteration, then mean
        return torch.stack(bag_probs).mean(dim=0)

    def _all_probs(self, X: pd.DataFrame) -> torch.Tensor:
        if self._kv_cache_mode():
            return self._predict_proba_cached(X)
        probs = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            pred = self._predict(X.iloc[start:end])
            probs.append(pred)
        return torch.cat(probs)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> list:
        """Predict the class labels for the provided input dataframe.

        Args:
            X: The input dataframe.

        Returns:
            The predicted class labels.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        probs = self._all_probs(X)

        preds = probs.argmax(dim=-1).numpy()
        return [self.classes_[p] for p in preds]

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the probabilities of the classes for the provided input dataframe.

        Args:
            X: The input data.

        Returns:
            The predicted probabilities of the classes.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        return self._all_probs(X).numpy()


class SAP_RPT_OSS_Regressor(RegressorMixin, SAP_RPT_OSS_Estimator):
    classification_or_regression = 'regression'

    def task_specific_fit(self):
        self.y_ = self.y_.astype(float)

    @torch.no_grad()
    def _predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the target variable.

        Args:
            X: The input dataframe.

        Returns:
            The predicted target variable.
        """
        # Check if fit has been called
        check_is_fitted(self)

        all_preds = []

        for bagging_index in range(self.bagging_number):
            tokenized_data = self.get_tokenized_data(X.copy(), bagging_index)
            tokenized_data = to_device(tokenized_data, self.device, raise_on_unexpected=False)
            logits_reg = self.model(**tokenized_data)
            label_classes = tokenized_data['label_classes']

            if self.regression_type != 'l2':
                if len(label_classes) != self.num_regression_bins:
                    raise ValueError(f'Expected {self.num_regression_bins} classes, got {len(label_classes)}')

            preds, _ = self.model.extract_prediction_regression(logits_reg,
                                                                tokenized_data['data']['target'],
                                                                tokenized_data['label_classes'],
                                                                target_mean=tokenized_data.get('target_mean'),
                                                                target_std=tokenized_data.get('target_std'))
            all_preds.append(preds)

        preds = np.mean(all_preds, axis=0)

        return preds

    @torch.no_grad()
    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """Predict the target variable.

        Args:
            X: The input dataframe.

        Returns:
            The predicted target variable.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.X_.columns)

        if self._kv_cache_mode():
            check_is_fitted(self)
            bag_preds = []
            for outputs in self.cached_bag_outputs(X):
                chunks = []
                for out, targets, label_classes, target_mean, target_std in outputs:
                    preds, _ = self.model.extract_prediction_regression(out, targets, label_classes,
                                                                        target_mean=target_mean,
                                                                        target_std=target_std)
                    chunks.append(np.atleast_1d(preds))
                bag_preds.append(np.concatenate(chunks))
            return np.mean(bag_preds, axis=0)

        preds = []
        for start in range(0, len(X), self.test_chunk_size):
            end = start + self.test_chunk_size
            pred = self._predict(X.iloc[start:end])
            preds.append(pred)
        preds = np.concatenate(preds)
        return preds

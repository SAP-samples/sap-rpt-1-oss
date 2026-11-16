# SPDX-FileCopyrightText: 2025 SAP SE
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Tuple

import torch
from torch import nn
from transformers.models.roberta.modeling_roberta import RobertaIntermediate, RobertaOutput, RobertaSelfOutput


class TwoDimensionalAttentionLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        layer_class = TorchRobertaLayer
        self.cross_column_layer = layer_class(config)
        self.cross_row_layer = layer_class(config)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: shape (num_rows, num_columns, hidden_size)
        attention_mask: shape (num_rows, num_rows) and values 0 (attend) or -inf (do not attend).
        Applies the two separate RobertaLayers, once along the "columns" direction and once along the "rows" direction.
        Along the columns direction, the attention is always full.
        Along the rows direction, an attention mask should be provided (to avoid that context rows can attend query rows)

        Returns: tensor of shape (num_rows, num_columns, hidden_size)
        """
        num_rows, num_columns, _ = hidden_states.shape

        # Cross-column attention it's easy: just treat row as distinct elements in the batch, so nothing to reshape
        horizontal_outputs = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        max_rows_per_batch = 8192
        col_fraction = 100.0 / float(num_columns)
        batch_step = int(max_rows_per_batch * col_fraction)

        for i in range(0, num_rows, batch_step):
            end_idx = i + batch_step
            chunk = hidden_states[i:end_idx, :, :]
            chunk_output = self.cross_column_layer(chunk)[0]
            horizontal_outputs[i:i + batch_step] = chunk_output

        # horizontal_outputs has shape (num_rows, num_columns, hidden_size)

        # For cross-row attention we can do the same, but we need to permute before and after.
        # Also, we use the attention mask here, if provided. But the attention mask has shape
        # (num_rows, num_rows) and we we unsqueeze to shape (1, 1, num_rows, num_rows)
        # (first two dims are batch size, which here is num_columns, and num_heads)
        batch_step = 100
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)
        horizontal_outputs = horizontal_outputs.transpose(0, 1).contiguous()

        vertical_outputs = torch.zeros_like(horizontal_outputs,
                                            dtype=horizontal_outputs.dtype,
                                            device=horizontal_outputs.device)
        for i in range(0, num_columns, batch_step):
            end_idx = i + batch_step
            chunk = horizontal_outputs[i:end_idx, :, :]
            chunk_output = self.cross_row_layer(chunk, attention_mask)[0]
            vertical_outputs[i:i + batch_step, :, :] = chunk_output

        return vertical_outputs.transpose(0, 1).contiguous()

    # ------------------------------------------------------------------
    # KV cache path. Context rows attend only to context rows, so their states never depend on the
    # query rows: their cross-row keys/values can be computed once and reused for every query batch.
    # ------------------------------------------------------------------
    # Cells (rows x columns) processed per sub-batch on the cached path. 8192 x 100 matches the
    # budget forward() uses; lower it to trade speed for memory on small GPUs.
    max_cells_per_batch = 8192 * 100

    def _cross_column(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_rows, num_columns, _ = hidden_states.shape
        outputs = torch.empty_like(hidden_states)
        step = max(1, self.max_cells_per_batch // num_columns)
        for i in range(0, num_rows, step):
            outputs[i:i + step] = self.cross_column_layer(hidden_states[i:i + step])[0]
        return outputs

    def forward_context(self, hidden_states: torch.Tensor, allocate, compute_output: bool = True):
        """
        Context rows only. hidden_states: (num_context, num_columns, hidden_size).
        allocate(shape, dtype) returns the empty tensor that receives the keys (and then the values); it decides
        where the cache lives (see ContextCache.empty).
        Returns (outputs or None, keys, values); keys/values have shape (num_columns, num_heads, num_context, head_dim).
        """
        num_rows, num_columns, _ = hidden_states.shape
        horizontal = self._cross_column(hidden_states).transpose(0, 1).contiguous()  # (cols, rows, hidden)
        self_attention = self.cross_row_layer.attention.self_attention
        kv_shape = (num_columns, self_attention.num_attention_heads, num_rows, self_attention.attention_head_size)
        keys = allocate(kv_shape, horizontal.dtype)
        values = allocate(kv_shape, horizontal.dtype)
        to_host = keys.device.type == 'cpu' and horizontal.is_cuda
        asynchronous = to_host and keys.is_pinned()
        vertical = torch.empty_like(horizontal) if compute_output else None
        step = max(1, self.max_cells_per_batch // num_rows)
        for i in range(0, num_columns, step):
            chunk = horizontal[i:i + step]
            key, value = self_attention.project_key_value(chunk)
            keys[i:i + step].copy_(key, non_blocking=asynchronous)
            values[i:i + step].copy_(value, non_blocking=asynchronous)
            if compute_output:
                # context attends to context only: no mask needed
                vertical[i:i + step] = self.cross_row_layer.attend(chunk, key, value, None)
        if asynchronous:
            torch.cuda.synchronize(horizontal.device)
        outputs = vertical.transpose(0, 1).contiguous() if compute_output else None
        return outputs, keys, values

    def forward_query(self, hidden_states: torch.Tensor, context_keys: torch.Tensor, context_values: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Query rows only. hidden_states: (num_query, num_columns, hidden_size); context_keys/values from
        forward_context; attention_mask: (1, 1, num_query, num_context + num_query), see RPT.build_query_attention_mask.
        Each query row attends to all context rows and to itself, exactly as in forward().
        """
        num_query, num_columns, _ = hidden_states.shape
        num_context = context_keys.shape[2]
        horizontal = self._cross_column(hidden_states).transpose(0, 1).contiguous()  # (cols, queries, hidden)
        self_attention = self.cross_row_layer.attention.self_attention
        vertical = torch.empty_like(horizontal)
        step = max(1, self.max_cells_per_batch // (num_context + num_query))
        device = horizontal.device
        for i in range(0, num_columns, step):
            chunk = horizontal[i:i + step]
            own_key, own_value = self_attention.project_key_value(chunk)
            key = torch.cat([context_keys[i:i + step].to(device, non_blocking=True), own_key], dim=2)
            value = torch.cat([context_values[i:i + step].to(device, non_blocking=True), own_value], dim=2)
            vertical[i:i + step] = self.cross_row_layer.attend(chunk, key, value, attention_mask)
        return vertical.transpose(0, 1).contiguous()


class TorchRobertaLayer(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.seq_len_dim = 1
        self.attention = TorchAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None) -> Tuple[torch.Tensor]:
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        # Return a tuple like the original RobertaLayer, though it's not very useful
        return (self.output(intermediate_output, attention_output),)

    def attend(self, hidden_states: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Same as forward, but attends to the given keys/values (batch, heads, seq_kv, head_dim)."""
        attention_output = self.attention.attend(hidden_states, key, value, attention_mask)
        return self.output(self.intermediate(attention_output), attention_output)


class TorchAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self_attention = TorchSelfAttention(config)
        self.output = RobertaSelfOutput(config)

    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None) -> Tuple[torch.Tensor]:
        self_outputs = self.self_attention(hidden_states, attention_mask)
        return self.output(self_outputs, hidden_states)

    def attend(self, hidden_states: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.output(self.self_attention.attend(hidden_states, key, value, attention_mask), hidden_states)


class TorchSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, 'embedding_size'):
            raise ValueError(f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                             f"heads ({config.num_attention_heads})")

        self.num_attention_heads = config.num_attention_heads
        assert config.hidden_size % config.num_attention_heads == 0, f'{config.hidden_size=} must be divisible by {config.num_attention_heads=}'
        self.attention_head_size = config.hidden_size // config.num_attention_heads

        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = config.attention_probs_dropout_prob

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch_size, num_heads, seq_len, head_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.FloatTensor] = None) -> torch.Tensor:
        """
        hidden_states: shape (batch_size, seq_len, hidden_size)
        attention_mask: shape (batch_size, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
        """
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False)

        context_layer = attn_output.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.config.hidden_size,)
        return context_layer.view(new_context_layer_shape)

    def project_key_value(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Keys and values of hidden_states (batch, seq, hidden), as (batch, heads, seq, head_dim)."""
        return (self.transpose_for_scores(self.key(hidden_states)),
                self.transpose_for_scores(self.value(hidden_states)))

    def attend(self, hidden_states: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
               attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Attention of hidden_states' queries over precomputed keys/values (batch, heads, seq_kv, head_dim)."""
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False)
        context_layer = attn_output.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.config.hidden_size,)
        return context_layer.view(new_context_layer_shape)

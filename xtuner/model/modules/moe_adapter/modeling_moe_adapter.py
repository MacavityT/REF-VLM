# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import warnings
from typing import List, Optional, Tuple, Union
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from torch.utils.checkpoint import checkpoint
from transformers.models.mixtral.modeling_mixtral import (
    load_balancing_loss_func,
    MixtralSparseMoeBlock,
    MixtralConfig
)

from .configuration_moe_adapter import MoEAdapterConfig

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MoeAdapterLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ffn=2048, dropout=0.1, num_experts=8, top_k=2):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.input_layernorm = nn.LayerNorm(d_model)
        self.post_attention_layernorm = nn.LayerNorm(d_model)

        moe_ffn_config = MixtralConfig(
            hidden_size=d_model,
            intermediate_size=d_ffn,
            num_local_experts=num_experts,
            num_experts_per_tok=top_k,
            hidden_act="silu",
        )
        self.sparse_moe_ffn = MixtralSparseMoeBlock(moe_ffn_config)

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
        residual = x
        x = self.input_layernorm(x)

        # Self Attention
        attn_output, _ = self.multihead_attn(
            x, 
            x, 
            x, 
            attn_mask=src_mask, 
            key_padding_mask=src_key_padding_mask
        )
        x = residual + attn_output

        # Fully Connected
        residual = x
        x = self.post_attention_layernorm(x)
        x, router_logits = self.sparse_moe_ffn(x)
        x = residual + x
        return x, router_logits

class MoEAdapterModel(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = MoEAdapterConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: MoEAdapterConfig):
        super().__init__(config)
        num_queries = config.num_queries
        d_input = config.d_input
        d_model = config.d_model
        n_heads = config.n_heads
        dropout = config.dropout
        d_ffn = config.d_ffn
        num_experts = config.num_experts
        top_k = config.top_k

        self.in_proj = nn.Linear(d_input, d_model)
        self.positional_encoding = PositionalEncoding(
            d_model, 
            dropout=0, 
            max_len=num_queries
        )
        self.layers = nn.ModuleList()
        for _ in range(config.num_layers):
            self.layers.append(
                MoeAdapterLayer(
                    d_model,
                    n_heads,
                    d_ffn,
                    dropout,
                    num_experts,
                    top_k
                )
            )
        self.last_norm = nn.LayerNorm(d_model)
        # Initialize weights and apply final processing
        self.post_init()

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)

        self.in_proj.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, MoEAdapterModel):
            module.gradient_checkpointing = value

    def compute_loss_moe(self, router_logits, attention_mask=None):
        moe_loss = load_balancing_loss_func(
            router_logits,
            self.config.num_experts,
            self.config.top_k,
            attention_mask,
        )
        return moe_loss

    def forward(self, 
        x, 
        attention_mask=None, 
        mode='loss'
    ):
        hidden_states = self.in_proj(x)
        hidden_states = self.positional_encoding(hidden_states)
        all_hidden_states = ()
        all_router_logits = ()
        for layer in self.layers:
            all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states, router_logits = checkpoint(layer, hidden_states, attention_mask)
            else:
                hidden_states, router_logits = layer(hidden_states, attention_mask)
            all_router_logits += (router_logits,)

        # add hidden states from the last decoder layer
        last_hidden_states = self.last_norm(hidden_states)
        all_hidden_states += (last_hidden_states,)

        loss = None
        if mode == 'loss':
            loss = load_balancing_loss_func(
            all_router_logits,
            self.config.num_experts,
            self.config.top_k,
            attention_mask,
        )
        return dict(
            loss = loss,
            hidden_states = all_hidden_states,
            all_router_logits = all_router_logits
        )

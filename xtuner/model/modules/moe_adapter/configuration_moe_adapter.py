# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig

class MoEAdapterConfig(PretrainedConfig):
    model_type = 'moe_adapter'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        num_queries=30,
        d_input=4096,
        d_model=4096,
        n_heads=8,
        dropout=0,
        d_ffn=1024,
        d_output=1024,
        num_experts=8,
        top_k=2,
        num_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_queries = num_queries
        self.d_input = d_input
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_ffn = d_ffn
        self.d_output = d_output
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_layers = num_layers 

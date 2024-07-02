# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig

class SyncTunerConfig(PretrainedConfig):
    model_type = 'sync_tuner'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        num_layers=3,
        num_queries=256,
        d_input=4096,
        d_model=4096,
        d_ffn=2048,
        output_dim=3,
        num_heads=8,
        dropout=0.1,
        ratio=0.5,
        **kwargs,
    ):
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.d_input = d_input
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.ratio = ratio
        super().__init__(**kwargs)


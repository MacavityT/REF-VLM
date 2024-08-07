# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig

class SyncTunerConfig(PretrainedConfig):
    model_type = 'sync_tuner'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        use_in_pred=True,
        num_layers=3,
        num_queries=256,
        d_input=4096,
        d_model=4096,
        d_ffn=2048,
        num_heads=8,
        dropout=0.1,
        ratio=0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_in_pred = use_in_pred
        self.num_layers = num_layers
        self.num_queries = num_queries
        self.d_input = d_input
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.num_heads = num_heads
        self.dropout = dropout
        self.ratio = ratio


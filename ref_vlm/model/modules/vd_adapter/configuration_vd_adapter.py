# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig

class VDAdapterConfig(PretrainedConfig):
    model_type = 'vd_adapter'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        phrase_max_length=100,
        ref_max_length=100,
        d_input_image=1024,
        d_input_token=4096,
        d_model=1024,
        n_heads=8,
        dropout=0,
        d_ffn=2048,
        num_layers=2,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # in fact, CLIP-text encoder only support max positional embedding as 77
        self.phrase_max_length = phrase_max_length
        self.ref_max_length = ref_max_length
        self.d_input_image = d_input_image
        self.d_input_token = d_input_token
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_ffn = d_ffn
        self.num_layers = num_layers
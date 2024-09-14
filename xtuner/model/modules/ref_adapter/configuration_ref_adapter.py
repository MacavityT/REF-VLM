# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig

class REFAdapterConfig(PretrainedConfig):
    model_type = 'ref_adapter'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        mode='encode',
        phrase_max_length=100,
        unit_max_length=50,
        ref_max_length=100,
        max_position_embedding=2048,
        d_input=4096,
        d_model=1024,
        n_heads=8,
        dropout=0,
        d_ffn=2048,
        num_layers=3,
        packing=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        assert mode in ['encode', 'decode']
        self.mode = mode
        # in fact, CLIP-text encoder only support max positional embedding as 77
        self.phrase_max_length = phrase_max_length
        self.unit_max_length = unit_max_length
        self.ref_max_length = ref_max_length
        self.max_position_embedding = max_position_embedding
        self.d_input = d_input
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = dropout
        self.d_ffn = d_ffn
        self.num_layers = num_layers
        self.packing = packing

# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class VPTEncoderConfig(PretrainedConfig):
    model_type = 'vpt_encoder'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        num_patches=9,
        patch_size=64,
        strategy='embedding',
        visual_hidden_size=1024,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.strategy = strategy
        assert strategy in ['embedding', 'pooling']
        super().__init__(**kwargs)

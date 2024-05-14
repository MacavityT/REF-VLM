# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class VPTEncoderConfig(PretrainedConfig):
    model_type = 'vpt_encoder'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        vis_feats_len=576,
        mask_patch_len=64,
        visual_hidden_size=1024,
        strategy='embedding',
        **kwargs,
    ):
        self.vis_feats_len = vis_feats_len
        self.mask_patch_len = mask_patch_len
        self.visual_hidden_size = visual_hidden_size
        self.strategy = strategy
        super().__init__(**kwargs)

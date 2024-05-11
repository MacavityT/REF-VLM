# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class VPTEncoderConfig(PretrainedConfig):
    model_type = 'vpt_encoder'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        # visual_hidden_size=4096,
        # llm_hidden_size=4096,
        # depth=2,
        # hidden_act='gelu',
        # bias=True,
        vis_feats_len=576,
        mask_patch_len=64,
        visual_hidden_size=1024,
        strategy='embedding',
        **kwargs,
    ):
        # self.visual_hidden_size = visual_hidden_size
        # self.llm_hidden_size = llm_hidden_size
        # self.depth = depth
        # self.hidden_act = hidden_act
        # self.bias = bias
        self.vis_feats_len = vis_feats_len
        self.mask_patch_len = mask_patch_len
        self.visual_hidden_size = visual_hidden_size
        self.strategy = strategy
        super().__init__(**kwargs)

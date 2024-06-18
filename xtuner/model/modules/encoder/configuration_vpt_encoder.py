# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class VPTEncoderConfig(PretrainedConfig):
    model_type = 'vpt_encoder'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        vis_feats_len=576,
        mask_patch_len=64,
        strategy='embedding',
        visual_hidden_size=1024,
        llm_hidden_size=4096,
        depth=2,
        hidden_act='gelu',
        bias=True,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.vis_feats_len = vis_feats_len
        self.mask_patch_len = mask_patch_len
        self.strategy = strategy

        # # projector
        # self.llm_hidden_size = llm_hidden_size
        # self.depth = depth
        # self.hidden_act = hidden_act
        # self.bias = bias
        super().__init__(**kwargs)

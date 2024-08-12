# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class VPTEncoderConfig(PretrainedConfig):
    model_type = 'vpt_encoder'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        num_patches=9,
        patch_size=8,
        strategy='pooling',
        visual_hidden_size=1024,
        use_mask_token=False,
        use_projector=False,
        llm_hidden_size=4096,
        hidden_act='gelu',
        depth=2,
        bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_hidden_size = visual_hidden_size
        self.num_patches = num_patches
        self.patch_size = patch_size
        self.strategy = strategy
        self.use_mask_token = use_mask_token
        self.use_projector = use_projector
        self.llm_hidden_size = llm_hidden_size
        self.hidden_act=hidden_act
        self.depth = depth
        self.bias = bias
        assert strategy in ['embedding', 'pooling']

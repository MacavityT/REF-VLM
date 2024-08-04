# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig


class MaskDecoderConfig(PretrainedConfig):
    model_type = 'mask_decoder'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        depth=2,
        hidden_act='gelu',
        bias=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.depth = depth
        self.hidden_act = hidden_act
        self.bias = bias
        

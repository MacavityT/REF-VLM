# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig
from .configuration_mask_decoder import MaskDecoderConfig

class DepthDecoderConfig(MaskDecoderConfig):
    model_type = 'depth_decoder'

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
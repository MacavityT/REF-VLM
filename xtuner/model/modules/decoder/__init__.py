# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_box_decoder import BoxDecoderConfig
from .modeling_box_decoder import BoxDecoderModel
from .configuration_mask_decoder import MaskDecoderConfig
from .modeling_mask_decoder import MaskDecoderModel

AutoConfig.register('box_decoder', BoxDecoderConfig)
AutoModel.register(BoxDecoderConfig, BoxDecoderModel)
AutoConfig.register('mask_decoder', MaskDecoderConfig)
AutoModel.register(MaskDecoderConfig, MaskDecoderModel)


__all__ = [
    'BoxDecoderConfig', 'BoxDecoderModel',
    'MaskDecoderConfig', 'MaskDecoderModel'
]

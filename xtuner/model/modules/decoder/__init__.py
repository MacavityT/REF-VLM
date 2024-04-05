# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_decoder import DecoderConfig
from .modeling_box_decoder import BoxDecoder

AutoConfig.register('decoder', DecoderConfig)
AutoModel.register(DecoderConfig, BoxDecoder)

__all__ = ['DecoderConfig', 'BoxDecoder']

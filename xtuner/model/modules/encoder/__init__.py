# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_vpt_encoder import VPTEncoderConfig
from .modeling_vpt_encoder import VPTEncoderModel

AutoConfig.register('vpt_encoder', VPTEncoderConfig)
AutoModel.register(VPTEncoderConfig, VPTEncoderModel)

__all__ = ['VPTEncoderConfig', 'VPTEncoderConfig']

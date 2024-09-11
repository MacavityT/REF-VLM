# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_vpt_encoder import VPTEncoderConfig
from .modeling_vpt_encoder import VPTEncoderModel
from .clip_convnext_encoder import CLIPConvNextModel

AutoConfig.register('vpt_encoder', VPTEncoderConfig)
AutoModel.register(VPTEncoderConfig, VPTEncoderModel)

__all__ = ['VPTEncoderConfig', 'VPTEncoderConfig', 'CLIPConvNextModel']

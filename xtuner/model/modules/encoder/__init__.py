# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_encoder import EncoderConfig
from .modeling_encoder import Encoder

AutoConfig.register('encoder', EncoderConfig)
AutoModel.register(EncoderConfig, Encoder)

__all__ = ['EncoderConfig', 'Encoder']

# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_moe_adapter import MoEAdapterConfig
from .modeling_moe_adapter import MoEAdapterModel

AutoConfig.register('moe_adapter', MoEAdapterConfig)
AutoModel.register(MoEAdapterConfig, MoEAdapterModel)

__all__ = ['MoEAdapterConfig', 'MoEAdapterModel']

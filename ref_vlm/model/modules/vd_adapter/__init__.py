# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_vd_adapter import VDAdapterConfig
from .modeling_vd_adapter import VDAdapterModel

AutoConfig.register('vd_adapter', VDAdapterConfig)
AutoModel.register(VDAdapterConfig, VDAdapterModel)

__all__ = ['VDAdapterConfig', 'VDAdapterModel']

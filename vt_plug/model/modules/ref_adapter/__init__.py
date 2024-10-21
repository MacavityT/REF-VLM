# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_ref_adapter import REFAdapterConfig
from .modeling_ref_adapter import REFAdapterModel

AutoConfig.register('ref_adapter', REFAdapterConfig)
AutoModel.register(REFAdapterConfig, REFAdapterModel)

__all__ = ['REFAdapterConfig', 'REFAdapterModel']

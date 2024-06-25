# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .configuration_sync_tuner import SyncTunerConfig
from .modeling_sync_tuner import SyncTunerModel

AutoConfig.register('sync_tuner', SyncTunerConfig)
AutoModel.register(SyncTunerConfig, SyncTunerModel)

__all__ = ['SyncTunerConfig', 'SyncTunerModel']

# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoConfig, AutoModel

from .modeling_decoder import DecoderModel
from .configuration_box_decoder import BoxDecoderConfig
from .modeling_box_decoder import BoxDecoderModel
from .configuration_mask_decoder import MaskDecoderConfig
from .modeling_mask_decoder import MaskDecoderModel
from .configuration_pose_decoder import PoseDecoderConfig
from .configuration_depth_decoder import DepthDecoderConfig
from .modeling_pose_decoder import PoseDecoderModel
from .modeling_depth_decoder import DepthDecoderModel

AutoConfig.register('box_decoder', BoxDecoderConfig)
AutoModel.register(BoxDecoderConfig, BoxDecoderModel)
AutoConfig.register('mask_decoder', MaskDecoderConfig)
AutoModel.register(MaskDecoderConfig, MaskDecoderModel)

AutoConfig.register('pose_decoder', PoseDecoderConfig)
AutoModel.register(PoseDecoderConfig, PoseDecoderModel)
AutoConfig.register('depth_decoder', DepthDecoderConfig)
AutoModel.register(DepthDecoderConfig, DepthDecoderModel)

__all__ = [
    'DecoderModel',
    'BoxDecoderConfig', 'BoxDecoderModel',
    'MaskDecoderConfig', 'MaskDecoderModel',
    'PoseDecoderConfig', 'PoseDecoderModel',
    'DepthDecoderConfig', 'DepthDecoderModel'
]

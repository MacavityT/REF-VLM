# Copyright (c) OpenMMLab. All rights reserved.
from .vt_map_fn import vt_map_fn
from .vt_map_fn_stage2 import (
    vt_map_fn_stage2,
    vt_keypoint_map_fn
)

__all__ = [
    'vt_map_fn',
    'vt_map_fn_stage2', 
    'vt_keypoint_map_fn'
]

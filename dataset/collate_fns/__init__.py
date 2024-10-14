# Copyright (c) OpenMMLab. All rights reserved.
from .default_collate_fn import default_collate_fn
from .vt_collate_fn import vt_collate_fn

__all__ = ['default_collate_fn', 'vt_collate_fn']

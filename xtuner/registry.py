# Copyright (c) OpenMMLab. All rights reserved.
from mmengine import DATASETS, TRANSFORMS, METRICS, FUNCTIONS, Registry

__all__ = ['BUILDER', 'MAP_FUNC', 'DATASETS', 'TRANSFORMS', 
           'METRICS', 'FUNCTIONS']

BUILDER = Registry('builder')
MAP_FUNC = Registry('map_fn')
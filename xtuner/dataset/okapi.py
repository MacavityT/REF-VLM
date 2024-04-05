# Copyright (c) OpenMMLab. All rights reserved.
import json
import logging
import os

import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset

from xtuner.registry import BUILDER
from .huggingface import process_hf_dataset
from .utils import expand2square

from .llava import LLaVADataset

# stage one with no special tokens, other stages add special tokens
STAGES = ['stage1', 'stage2', 'stage3', 'stage4']

class OkapiDataset(LLaVADataset):

    def __init__(self, stage, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        assert stage in STAGES
        self.stage = stage


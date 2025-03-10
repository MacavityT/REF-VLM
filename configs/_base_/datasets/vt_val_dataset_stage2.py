# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from ref_vlm.dataset.map_fns import (
    vt_map_fn_stage2,
    vt_template_map_fn_factory
)
from ref_vlm.dataset import VTInstructDataset
from ref_vlm.dataset.collate_fns import vt_collate_fn
from mmengine.dataset import DefaultSampler

with read_base():
    from .train_all_dataset import train_all_dataset
    from ..models.all_tokenizers import vicuna_7b_path_tokenizer
    from ..models.ref_vlm_encoders import clip_patch14_336

from ref_vlm.utils import PROMPT_TEMPLATE
prompt_template = PROMPT_TEMPLATE.vd_cot

# Data configs
max_length = 2048 - 576 # use cutoff lens instead
cutoff_len = 2048
visual_hidden_size = 1024 # visual_encoder.config.hidden_size
vpt_num_patches = 9
vpt_patch_size = 8 # sqrt(576/9)=8

# Data
val_cfg = dict(type='ValLoop')

val_all_dataset = dict(
    caption=dict(
        type='CaptionDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CAP_coco2017_val.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/val2017',
        stage=2,
        template_name=r'image_cap',
    ),
    vqav2_val=dict(
        type='VQAv2Dataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/v2_OpenEnded_mscoco_val2014_questions.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/VQAv2/real_images/',
        stage=2,
        template_name=r"VQA",
    ),
)








# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from mmengine.dataset import DefaultSampler

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.dataset import LLaVADataset, ConcatDataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory

from vt_plug.dataset.map_fns import vt_map_fn
from vt_plug.dataset import VTInstructDataset
from vt_plug.evaluation.metrics import ImgCapComputeMetrics

with read_base():
    from .train_all_dataset import train_all_dataset
    from ..models.all_tokenizers import vicuna_7b_path_tokenizer
    from ..models.all_visual_encoders import clip_patch14_336



# Data
val_cfg = dict(type='ValLoop')

val_all_dataset = dict(
    caption=dict(
        type='CaptionDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CAP_coco2017_val.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/val2017',
        template_name=r'image_cap',
    ),
    vqav2_val=dict(
        type='VQAv2Dataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/v2_OpenEnded_mscoco_val2014_questions.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/VQAv2/real_images/',
        template_name=r"VQA",
    ),
)






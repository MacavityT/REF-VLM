from mmengine.config import read_base
from mmengine.dataset import DefaultSampler

from xtuner.utils import PROMPT_TEMPLATE
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.dataset import LLaVADataset, ConcatDataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory

from ref_vlm.dataset.map_fns import vt_map_fn
from ref_vlm.dataset import VTInstructDataset
from ref_vlm.evaluation.metrics import ImgCapComputeMetrics, VQAComputeMetrics

with read_base():
    from .train_all_dataset import train_all_dataset
    from .test_reg_variant import test_reg_variant
    from ..models.all_tokenizers import vicuna_7b_path_tokenizer
    from ..models.ref_vlm_encoders import clip_patch14_336
    

# Params
prompt_template = PROMPT_TEMPLATE.vicuna
max_length = int(2048 - (336 / 14)**2)
cutoff_len = 2048
visual_hidden_size = 1024

# Datasets
test_cfg = dict(type='TestLoop')

test_all_dataset = dict(
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
    reg=dict(
        type='REGDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_refcocog_umd_test.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        template_name=r'REG',
        map_placeholders=dict(
            input=["<boxes>"],
        )        
    ),
    **test_reg_variant,
)

test_dataset_args = [
    dict(
        type='SubSet',
        portion=1/20,
        do_shuffle=True,
        seed=43,
        cfg=test_all_dataset['vqav2_val'],
            )
    
]

test_evaluator = dict(
    type=VQAComputeMetrics, tokenizer=vicuna_7b_path_tokenizer, prefix='vqa')
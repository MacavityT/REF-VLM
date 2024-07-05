from mmengine.dataset import DefaultSampler
from xtuner.dataset import OkapiDataset,ConcatDataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory, okapi_map_fn
from xtuner.utils import PROMPT_TEMPLATE
from mmengine.config import read_base
from xtuner.evaluation.metrics.single_metric import ImgCapComputeMetrics,VQAComputeMetrics

with read_base():
    from ..models.all_tokenizers import vicuna_7b_path_tokenizer
    from ..models.all_visual_encoders import clip_patch14_336
    from .test_reg_variant import test_reg_variant

# Data
prompt_template = PROMPT_TEMPLATE.vicuna
max_length = int(2048 - (336 / 14)**2)


dataloader_num_workers = 5
test_cfg = dict(type='TestLoop')

test_all_dataset = dict(
    caption=dict(
        type='CaptionDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CAP_coco2017_val.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/val2017',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2017_val_shape.jsonl',
        template_name=r'image_cap',
    ),
    vqav2_val=dict(
        type='VQAv2Dataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/v2_OpenEnded_mscoco_val2014_questions.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/VQAv2/real_images/',
        image_info_folder='/data/Aaronzhu/DatasetStage1/Shikra/shape/vqav2_val_shape.jsonl',
        template_name=r"VQA",
    ),
    reg=dict(
        type='REGDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_refcocog_umd_test.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2014_train_shape.jsonl',
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
        enforce_online=True,
        cfg=test_all_dataset['vqav2_val'],
            )
    
]





test_evaluator = dict(
    type=VQAComputeMetrics, tokenizer=vicuna_7b_path_tokenizer, prefix='vqa')

# test_evaluator = dict(
#     type=ImgCapComputeMetrics, tokenizer=vicuna_7b_path_tokenizer, prefix='caption')






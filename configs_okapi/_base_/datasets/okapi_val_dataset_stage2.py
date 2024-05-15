from mmengine.dataset import DefaultSampler
from xtuner.dataset import OkapiDataset,ConcatDataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory, okapi_map_fn
from mmengine.config import read_base
from xtuner.evaluation.metrics.single_metric import ImgCapComputeMetrics,VQAComputeMetrics

with read_base():
    from ..models.all_tokenizers import vicuna_7b_path_tokenizer
    from ..models.all_visual_encoders import clip_patch14_336


# Data

val_cfg = dict(type='ValLoop')

val_all_dataset = dict(
    caption=dict(
        type='CaptionDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CAP_coco2017_val.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/val2017',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2017_val_shape.jsonl',
        stage=2,
        template_name=r'image_cap',
    ),
    vqav2_val=dict(
        type='VQAv2Dataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/v2_OpenEnded_mscoco_val2014_questions.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/VQAv2/real_images/',
        image_info_folder='/data/Aaronzhu/DatasetStage1/Shikra/shape/vqav2_val_shape.jsonl',
        stage=2,
        template_name=r"VQA",
    ),
)

val_dataset_args = [
    dict(
        type='SubSet',
        portion=1/20,
        do_shuffle=True,
        seed=43,
        enforce_online=True,
        cfg=val_all_dataset['caption'],
            )
    
]






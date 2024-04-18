from mmengine.dataset import DefaultSampler
from xtuner.dataset import OkapiDataset,ConcatDataset
from xtuner.dataset.collate_fns import default_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.utils import PROMPT_TEMPLATE
from mmengine.config import read_base
from xtuner.evaluation.metrics.single_metric import ImgCapComputeMetrics

with read_base():
    from .train_all_dataset import train_all_dataset
    from ..models.all_tokenizers import vicuna_7b_path_tokenizer
    from ..models.all_visual_encoders import clip_patch14_336


# Data
prompt_template = PROMPT_TEMPLATE.vicuna
max_length = int(2048 - (336 / 14)**2)


dataloader_num_workers = 5
val_cfg = dict(type='ValLoop')

val_all_dataset = dict(
    caption=dict(
        type='CaptionDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CAP_coco2017_val.jsonl',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/val2017',
        image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/coco2017_val_shape.jsonl',
        template_name=r'image_cap',
    )
)

val_dataset_args = [
    dict(
        type='SubSet',
        portion=1/10,
        do_shuffle=True,
        seed=43,
        cfg=val_all_dataset['caption'],
            )
    
]


okapi_dataset_val = dict(
    type=OkapiDataset,
    dataset=val_dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    tokenizer=vicuna_7b_path_tokenizer,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

val_dataloader = dict(
    batch_size=1,
    num_workers=dataloader_num_workers,
    dataset=okapi_dataset_val,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))


val_evaluator = dict(
    type=ImgCapComputeMetrics, tokenizer=vicuna_7b_path_tokenizer, prefix='caption')







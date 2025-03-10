# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base

with read_base():
    from ._base_.models.ref_vlm_encoders import clip_patch14_336
    from ._base_.datasets.vt_train_dataset_stage1 import *
    from ._base_.datasets.vt_val_dataset_stage1 import *
    from ._base_.models.ref_vlm_vicuna_7b import *
    from ._base_.schedules.schedule import *
    from ._base_.default_runtime import *

# Data configs
batch_size = 32  # per_device
dataloader_num_workers = 5

train_dataset = dict(
    type=VTInstructDataset,
    dataset=dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=vt_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

#region llava dataset
llava_dataset = dict(
    type=LLaVADataset,
    data_path=r"/data/Aaronzhu/DatasetStage1/llava/llava-pretrain/LLaVA-Pretrain/blip_laion_cc_sbu_558k_filter_new.json",
    image_folder=r'/data/Aaronzhu/DatasetStage1/llava/llava-pretrain/LLaVA-Pretrain/images',
    tokenizer=tokenizer,
    image_processor=clip_patch14_336['image_processor'],
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)
#endregion

# train_dataset = dict(type=ConcatDataset, datasets=[llava_dataset, train_dataset])

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))

val_dataset = dict(
    type=VTInstructDataset,
    dataset=val_dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=vt_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

val_dataloader = dict(
    batch_size=1,
    num_workers=dataloader_num_workers,
    dataset=val_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=default_collate_fn))


val_evaluator = dict(
    type=ImgCapComputeMetrics, tokenizer=tokenizer, prefix='caption')

# val_evaluator = dict(
#     type=VQAComputeMetrics, tokenizer=tokenizer, prefix='vqa')


# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=tokenizer)
]
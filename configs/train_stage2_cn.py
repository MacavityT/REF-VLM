# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.config import read_base
from vt_plug.dataset.map_fns import (
    vt_template_map_fn_factory
)

with read_base():
    from ._base_.models.all_visual_encoders import clip_patch14_336, clip_big14_224
    from ._base_.datasets.vt_train_dataset_stage2_cn import *
    from ._base_.datasets.vt_val_dataset_stage1 import *
    from ._base_.models.vt_plug_qwen import *
    from ._base_.schedules.schedule import *
    from ._base_.default_runtime import *

# Data configs
batch_size = 10  # per_device
dataloader_num_workers = 5

# Params
max_length = int(2048 - (336 / 14)**2)
cutoff_len = 2048
visual_hidden_size = 1024
model_dir = '/code/VT-PLUG/checkpoints/Qwen2.5/xtuner_output/stage1/0102_instruct/iter_118989.pth'

val_dataset_args = [
    dict(
        type='SubSet',
        portion=1/20,
        do_shuffle=True,
        seed=43,
        cfg=val_all_dataset['caption'],
    )
]


train_dataset = dict(
    type=VTInstructDataset,
    dataset=dataset_args,
    # image_processor=clip_patch14_336['image_processor'],
    image_processor=clip_big14_224['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=vt_map_fn,
    template_map_fn=dict(
        type=vt_template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)



train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))

val_dataset = dict(
    type=VTInstructDataset,
    dataset=val_dataset_args,
    # image_processor=clip_patch14_336['image_processor'],
    image_processor=clip_big14_224['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=vt_map_fn,
    template_map_fn=dict(
        type=vt_template_map_fn_factory, template=prompt_template),
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

model = dict(
    type=VTPlugModel,
    freeze_llm=False,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    pretrained_pth=model_dir,
    stage=1,
    projector=dict(
        llm_hidden_size=3584,
        depth=2,
    ),
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=qwen2_7b_path,
        trust_remote_code=True),
    # visual_encoder=clip_patch14_336['visual_encoder'],
    visual_encoder=clip_big14_224['visual_encoder'],
    loss_coefficient=dict(llm=1.),
    )

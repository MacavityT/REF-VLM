# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoModel
from transformers.utils.quantization_config import BitsAndBytesConfig
import torch
from mmengine.config import read_base
from peft import LoraConfig
with read_base():
    from .._base_.models.ref_vlm_encoders import *
    from .._base_.datasets.vt_train_dataset_stage2 import *
    from .._base_.datasets.vt_val_dataset_stage2 import *
    from .._base_.models.ref_vlm_vicuna_7b import *
    from .._base_.schedules.schedule import *
    from .._base_.default_runtime import *


# Data configs
batch_size = 16  # per_device
dataloader_num_workers = 8

# dataset reg
dataset_args = [
    train_all_dataset['reg'],
]
for dataset in dataset_args:
    if dataset['type'] == 'SubSet':
        dataset['cfg'].setdefault('stage',2)
    else:
        dataset['stage'] = 2

train_dataset = dict(
    type=VTInstructDataset,
    dataset=dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=dict(
        function=vt_map_fn_stage2,
    ),
    template_map_fn=dict(
        type=vt_template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True, seed=42),
    collate_fn=dict(type=vt_collate_fn))

val_cfg = None

# config models
pretrained_pth = 'checkpoints/vicuna_7b/stage1/0510_1_20_gc_rvg/iter_3558.pth'

model=dict(
    type=REFVLMModel,
    pretrained_pth=pretrained_pth,
    freeze_llm=False,
    freeze_visual_encoder=True,
    tokenizer=tokenizer,
    cutoff_len=cutoff_len,
    visual_encoder=clip_patch14_336['visual_encoder'],
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=vicuna_7b_path,
        trust_remote_code=True,
    ),
    vpt_encoder=dict(
        strategy='pooling',
        patch_size=vpt_patch_size,
        num_patches = vpt_num_patches,
        visual_hidden_size=visual_hidden_size,
        use_mask_token=False,
        use_projector=False,
        legacy=False
    ),
    loss_coefficient=dict(
        llm=1,
    )
)
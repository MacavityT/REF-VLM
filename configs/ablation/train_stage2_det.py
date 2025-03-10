# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoModel
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
batch_size = 15  # per_device
dataloader_num_workers = 4
max_epochs = 5

# dataset grand det and seg
dataset_args = [
    train_all_dataset['coco_rem_box'],
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
        args = dict(
            use_cot=True,
        )
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
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    visual_encoder=clip_patch14_336['visual_encoder'],
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=vicuna_7b_path,
        trust_remote_code=True,
    ),
    visual_decoder=dict(
        box=dict(
            use_group_matcher=False,
            num_queries=100,
            # quries_input_dim=256,
            quries_input_dim=4096,
            encoder_input_transform='resize_concat',
            # encoder_input_dim shape = [[16, 16, 1024], [32, 32, 1024], [64, 64, 1024]]
            encoder_input_index=[8, 16, 23], # clip-vit features
            encoder_input_dim=[1024, 1024, 1024],

            # encoder_input_index=[0, 1, 2, 3], # clip-convnext features
            # encoder_input_dim=[192, 384, 768, 1536],

            # encoder_input_index=[0, 1, 2, 4], # clip-convnext features with clip-vpt features
            # encoder_input_dim=[192, 384, 768, 1024],

            decoder_layers=6,
            decoder_ffn_dim=2048,
            decoder_attention_heads=8,
            decoder_layerdrop=0.0,
            activation_function="relu",
            d_model=256,
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            bbox_loss_coefficient=5,
            giou_loss_coefficient=2,
        ),
    ),
    loss_coefficient=dict(
        llm=1,
        box=1
    ))
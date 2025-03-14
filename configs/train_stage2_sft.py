# Copyright (c) OpenMMLab. All rights reserved.
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from xtuner.engine.runner import TrainLoop
from transformers import AutoModel
from mmengine.config import read_base

with read_base():
    from ._base_.models.ref_vlm_encoders import clip_patch14_336,clip_convnext_512
    from ._base_.datasets.vt_train_dataset_stage2 import *
    from ._base_.datasets.vt_val_dataset_stage2 import *
    from ._base_.datasets.vt_test_dataset_stage2 import *
    from ._base_.models.ref_vlm_vicuna_7b import *
    from ._base_.default_runtime import *


# Data configs
batch_size = 16  # per_device
dataloader_num_workers = 8
accumulative_counts = 1

max_epochs = 3
lr = 2e-5 # 2e-5 4e-6 2e-6
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.5

model_dir = "checkpoints/vicuna_7b/hf_model/1113_rec_refcoco_iter18860"


dataset_args_sft = [
    # train_all_dataset['lvis_box']
    # train_all_dataset['coco_rem_box']
    # train_all_dataset['train_cityscapes_instance'],
    # train_all_dataset['train_ade20_with_instance'],
    # train_all_dataset['interact_mask_finetune']
    # train_all_dataset['interact_scribble_finetune']
    # train_all_dataset['interact_point_finetune']
    # train_all_dataset['interact_box_finetune'],
    # train_all_dataset['point_local_b'],
    # train_all_dataset['point_local_p'],
    # train_all_dataset['point_twice_oq_bp'],
    # train_all_dataset['point_twice_sq_bp'],
    # train_all_dataset['point_twice_gq_bp'],
    # train_all_dataset['point_v7w_p'],
    # train_all_dataset['point_v7w_b'],
    # train_all_dataset['openpsg'],
    # train_all_dataset['train_ade20_with_instance'],
    # train_all_dataset['train_cityscapes_instance'],
    # train_all_dataset['coco_rem_mask'],
    # train_all_dataset['rec']
    # train_all_dataset['res_refcoco'],
    # train_all_dataset['res_refcocoa'],
    # train_all_dataset['res_refcocog'],
    # train_all_dataset['rec_refcoco'],
    # train_all_dataset['rec_refcocog'],
    # train_all_dataset['reg_refcocog_train_mask'],
    # train_all_dataset['flickr_caption'],
    # train_all_dataset['caption'],
    # train_all_dataset['vqav2_train'],
    # train_all_dataset['vqae_train'],
    # train_all_dataset['vqax_train'],
]

for dataset in dataset_args_sft:
    if dataset['type'] == 'SubSet':
        dataset['cfg'].setdefault('stage',2)
    else:
        dataset['stage'] = 2

# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=AdamW, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
# param_scheduler = [
#     dict(
#         type=LinearLR,
#         start_factor=1e-5,
#         by_epoch=True,
#         begin=0,
#         end=warmup_ratio * max_epochs,
#         convert_to_iter_based=True),
#     dict(
#         type=CosineAnnealingLR,
#         eta_min=0.0,
#         by_epoch=True,
#         begin=warmup_ratio * max_epochs,
#         end=max_epochs,
#         convert_to_iter_based=True)
# ]

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=max_epochs,
        convert_to_iter_based=True),
]


# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs,val_interval=500)




train_dataset = dict(
    type=VTInstructDataset,
    dataset=dataset_args_sft,
    image_processor=clip_patch14_336['image_processor'],
    image_tower_processor=clip_convnext_512['image_processor'],
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
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=vt_collate_fn))

val_cfg = None

# config models

llm=dict(
    type=AutoModelForCausalLM.from_pretrained,
    pretrained_model_name_or_path=model_dir,
    trust_remote_code=True)

projector = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=f'{model_dir}/projector',
    trust_remote_code=True,
)

vpt_encoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=f'{model_dir}/vpt_encoder',
    trust_remote_code=True,
)

box_decoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=f'{model_dir}/box_decoder',
    trust_remote_code=True,
)

mask_decoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=f'{model_dir}/mask_decoder',
    trust_remote_code=True,
)


model = dict(
    type=REFVLMModel,
    freeze_llm=False,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    llm=llm,
    visual_encoder=clip_patch14_336['visual_encoder'],
    visual_tower=clip_convnext_512['visual_encoder'],
    projector=projector,
    vpt_encoder=vpt_encoder,
    visual_decoder=dict(
        box=box_decoder,
        mask=mask_decoder
    ),
    loss_coefficient=dict(
        llm=1,
        box=0.5,
        mask=0.5
    )
)
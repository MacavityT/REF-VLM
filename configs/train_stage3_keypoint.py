# Copyright (c) OpenMMLab. All rights reserved.

from mmengine.optim import AmpOptimWrapper, LinearLR
from torch.optim import AdamW
from xtuner.engine.runner import TrainLoop
from dataset.collate_fns import vt_collate_fn
from transformers import AutoModel
from mmengine.config import read_base
with read_base():
    from ._base_.models.all_visual_encoders import clip_patch14_336, clip_convnext_512
    from ._base_.datasets.vt_train_dataset_stage2 import *
    from ._base_.datasets.vt_val_dataset_stage2 import *
    from ._base_.models.vt_plug_vicuna_7b import *
    from ._base_.default_runtime import *

# Data configs
batch_size = 8  # per_device
dataloader_num_workers = 4
accumulative_counts = 1
max_epochs = 20
lr = 1e-4 # 2e-5 4e-6 2e-6
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.5

model_dir = "checkpoints/vicuna_7b/hf_model/0929_keypoint_iter14000"

dataset_args_sft = [
    train_all_dataset['keypoints2017_det'],
    train_all_dataset['keypoints2014_det'],
]

for dataset in dataset_args_sft:
    if dataset['type'] == 'SubSet':
        dataset['cfg'].setdefault('stage', 2)
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
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs, val_interval=500)

train_dataset = dict(
    type=VTInstructDataset,
    dataset=dataset_args_sft,
    image_processor=clip_patch14_336['image_processor'],
    image_tower_processor=clip_convnext_512['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=dict(
        function=vt_keypoint_map_fn,
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

pose_decoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=f'{model_dir}/pose_decoder',
    trust_remote_code=True,
)

mask_decoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=f'{model_dir}/mask_decoder',
    trust_remote_code=True,
)

model=dict(
    type=VTPlugModel,
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
        pose=pose_decoder
        # pose=dict(
        #     num_queries=100,
        #     encoder_input_transform='resize_concat',
        #     encoder_input_index=[0, 1, 2, 4], # clip-convnext features with clip-vpt features
        #     encoder_input_dim=[192, 384, 768, 1024],
        #     use_group_matcher=True,  # True
        #     use_auxiliary_loss=False,
        #     aux_loss_coefficient=0.5,
        #     # box_config=dict(
        #     #     quries_input_dim=4096,
        #     #     decoder_layers=6,
        #     #     decoder_ffn_dim=2048,
        #     #     decoder_attention_heads=8,
        #     #     decoder_layerdrop=0.0,
        #     #     activation_function="relu",
        #     #     d_model=256,
        #     #     dropout=0.1,
        #     #     attention_dropout=0.0,
        #     #     activation_dropout=0.0,
        #     #     bbox_loss_coefficient=5, # 5
        #     #     giou_loss_coefficient=2, # 2
        #     # ),
        #     box_config=box_decoder,
        #     keypoint_config=dict(
        #         quries_input_dim=256,
        #         decoder_layers=6,
        #         decoder_ffn_dim=2048,
        #         decoder_attention_heads=8,
        #         decoder_layerdrop=0.0,
        #         activation_function="relu",
        #         d_model=256,
        #         dropout=0.1,
        #         attention_dropout=0.0,
        #         activation_dropout=0.0,
        #         num_body_points=17,
        #         keypoint_loss_coefficient=2,  #2
        #         oks_loss_coefficient=2,  #2
        #         cls_loss_coefficient=1,  #1
        #     )
        # )
    ),
    loss_coefficient=dict(
        llm=1,
        pose=1
    )
)
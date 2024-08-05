# Copyright (c) OpenMMLab. All rights reserved.

from functools import partial
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.dataset.map_fns import (
    okapi_map_fn_stage2,
    okapi_template_map_fn_factory
)
from xtuner.dataset.collate_fns import okapi_collate_fn
from transformers import AutoModel
from mmengine.config import read_base
with read_base():
    from ._base_.models.all_visual_encoders import clip_patch14_336
    from ._base_.datasets.okapi_train_dataset_stage2 import *
    # from ._base_.datasets.sketch_train_dataset_stage2 import *
    from ._base_.datasets.okapi_val_dataset_stage2 import *
    from ._base_.models.okapi_vicuna_7b import *
    # from ._base_.models.okapi_llama3_8b import *
    # from ._base_.models.okapi_mistral_7b import *
    from ._base_.default_runtime import *


# Data configs
max_length = 2048 - 576 # use cutoff lens instead  4096 
cutoff_len = 2048
visual_hidden_size = 1024 # visual_encoder.config.hidden_size
batch_size = 15  # per_device
dataloader_num_workers = 4
vrt_length = 256
vpt_num_patches = 9
vpt_patch_size = 8 # sqrt(576/9)=8
ref_length = 1
prompt_template = PROMPT_TEMPLATE.okapi
cot_weight = 1
vrt_weight = 1
accumulative_counts = 1

max_epochs = 2
lr = 2e-5  # 2e-5 4e-6 2e-6
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.5

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
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs,val_interval=500)


dataset_args_sft = train_all_dataset['reg']
dataset_args_sft['stage'] = 2
dataset_args_sft = [dataset_args_sft]



train_dataset = dict(
    type=OkapiDataset,
    dataset=dataset_args_sft,
    image_processor=clip_patch14_336['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=dict(
        function=okapi_map_fn_stage2,
        args = dict(
            vrt_len=vrt_length, 
            ref_len=ref_length
        )
    ),
    template_map_fn=dict(
        type=okapi_template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=okapi_collate_fn))

val_cfg = None

# config models

projector = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path='/code/okapi-mllm/sketch_checkpoints/0719_iter59525/projector',
    trust_remote_code=True,
)

vpt_encoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path='/code/okapi-mllm/sketch_checkpoints/0719_iter59525/vpt_encoder',
    trust_remote_code=True,
)

visual_sync_tuner = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path='/code/okapi-mllm/sketch_checkpoints/0719_iter59525/visual_sync_tuner',
    trust_remote_code=True,
)

model = dict(
    type=OkapiModel,
    freeze_llm=False,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path='/code/okapi-mllm/sketch_checkpoints/0719_iter59525',
        trust_remote_code=True),
    visual_encoder=clip_patch14_336['visual_encoder'],
    projector=projector,
    vpt_encoder=vpt_encoder,
    visual_sync_tuner=visual_sync_tuner,
    cot_weight=cot_weight,
    vrt_weight=vrt_weight)
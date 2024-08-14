# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.dataset.map_fns import (
    okapi_map_fn_stage2,
    okapi_template_map_fn_factory
)
from xtuner.dataset.collate_fns import okapi_collate_fn

from mmengine.config import read_base
with read_base():
    from ._base_.models.all_visual_encoders import clip_patch14_336
    from ._base_.datasets.okapi_train_dataset_stage2 import *
    # from ._base_.datasets.sketch_train_dataset_stage2 import *
    from ._base_.datasets.okapi_val_dataset_stage2 import *
    from ._base_.models.okapi_vicuna_7b import *
    # from ._base_.models.okapi_llama3_8b import *
    # from ._base_.models.okapi_mistral_7b import *
    from ._base_.schedules.schedule import *
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

train_dataset = dict(
    type=OkapiDataset,
    dataset=dataset_args,
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
pretrained_pth = '/model/Aaronzhu/OkapiModel/vicuna_7b/stage1/0510_1_20_gc_rvg/iter_3558.pth'

model = dict(
    type=OkapiModel,
    pretrained_pth=pretrained_pth,
    freeze_llm=False,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=vicuna_7b_path,
        trust_remote_code=True),
    visual_encoder=clip_patch14_336['visual_encoder'],
    vpt_encoder=dict(
        strategy='pooling',
        patch_size=vpt_patch_size,
        num_patches = vpt_num_patches,
        visual_hidden_size=visual_hidden_size,
        use_mask_token=False,
        use_projector=False
    ),
    visual_sync_tuner=dict(
        use_in_pred=True,
        num_layers=3,
        num_queries=vrt_length,
        d_input=4096,
        d_model=512,
        d_ffn=2048,
        num_heads=8,
        dropout=0.1,
        ratio=0.5
    ),
    loss_coefficient=dict(
        llm=1.,
        rec=0.5,
        moe=0.02,
        box=0.5,
        mask=0.5
    )
)
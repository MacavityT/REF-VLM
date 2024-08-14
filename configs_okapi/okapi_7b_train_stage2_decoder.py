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
    from ._base_.datasets.okapi_val_dataset_stage2 import *
    from ._base_.models.okapi_vicuna_7b import *
    from ._base_.schedules.schedule import *
    from ._base_.default_runtime import *



# Data configs
max_length = 2048 - 576 # use cutoff lens instead
cutoff_len = 2048
visual_hidden_size = 1024 # visual_encoder.config.hidden_size
batch_size = 2  # per_device
dataloader_num_workers = 1
vrt_length = 256
vpt_num_patches = 9
vpt_patch_size = 8 # sqrt(576/9)=8
ref_length = 1
ref_max_num = 30
ref_num_queries = ref_max_num * ref_length
prompt_template = PROMPT_TEMPLATE.okapi


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
    sampler=dict(type=DefaultSampler, shuffle=True, seed=42),
    collate_fn=dict(type=okapi_collate_fn))

val_cfg = None

# config models
pretrained_pth = '/model/Aaronzhu/OkapiModel/vicuna_7b/stage1/0510_1_20_gc_rvg/iter_3558.pth'

model=dict(
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
        use_mask_token=True,
        use_projector=False
    ),
    visual_sync_tuner=dict(
        use_in_pred=True,
        num_layers=3,
        num_queries=vrt_length,
        d_input=4096,
        d_model=1024,
        d_ffn=2048,
        num_heads=8,
        dropout=0.1,
        ratio=0.5
    ),
    # moe_adapter=dict(
    #     num_queries=ref_num_queries,
    #     d_input=4096,
    #     d_model=256,
    #     n_heads=8,
    #     dropout=0,
    #     d_ffn=2048,
    #     num_experts=8,
    #     top_k=2,
    #     num_layers=3,
    # ),
    visual_decoder=dict(
        box=dict(
            num_queries=ref_num_queries,
            # quries_input_dim=256,
            quries_input_dim=4096,
            encoder_input_transform='resize_concat',
            # encoder_input_dim shape = [[16, 16, 1024], [32, 32, 1024], [64, 64, 1024]]
            encoder_input_index=[3, 2, 1], 
            encoder_input_dim=[1024, 1024, 1024],
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
        mask=dict(
            num_queries=ref_num_queries,
            # quries_input_dim=256,
            quries_input_dim=4096,
            encoder_input_transform='multiple_select',
            # encoder_input_dim shape = [[16, 16, 1024], [32, 32, 1024], [64, 64, 1024]]
            encoder_input_index=[3, 2, 1], 
            encoder_input_dim=[1024, 1024, 1024],
            #region query decoder config
            decoder_layers=6,
            decoder_ffn_dim=2048,
            decoder_attention_heads=8,
            decoder_layerdrop=0.0,
            pre_norm=False,
            activation_function="relu",
            d_model=256,
            dropout=0.0,
            attention_dropout=0.0,
            activation_dropout=0.0,
            #endregion
            #region pixel decoder config
            encoder_layers=6, 
            fpn_feature_size=256,
            mask_feature_size=256,
            feature_strides=[4, 8, 16, 32],
            common_stride=4,
            encoder_feedforward_dim=1024,
            mask_loss_coefficient=20,
            dice_loss_coefficient=1,
            #endregion
        ),
    ),
    loss_coefficient=dict(
        llm=1,
        rec=0.5,
        moe=0.02,
        box=0.5,
        mask=0.5
    ))
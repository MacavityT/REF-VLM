# Copyright (c) OpenMMLab. All rights reserved.
from functools import partial
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.dataset.map_fns import (
    okapi_map_fn_stage2,
    okapi_template_map_fn_factory
)
# import os
from transformers import AutoModel
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
batch_size = 15  # per_device  24
dataloader_num_workers = 4
vrt_length = 0  # 256
vpt_num_patches = 9
vpt_patch_size = 8 # sqrt(576/9)=8
ref_length = 1
ref_box_num = 100
ref_mask_num = 30
ref_box_queries = ref_box_num * ref_length
ref_mask_queries = ref_mask_num * ref_length
prompt_template = PROMPT_TEMPLATE.okapi


# dataset grand det and seg
dataset_args = [
    train_all_dataset['flickr'],
    train_all_dataset['rec'],
    train_all_dataset['res_refcoco'],
    train_all_dataset['res_refcocoa'],
    train_all_dataset['res_refcocog'],
    train_all_dataset['llavag_gcg'],
    train_all_dataset['openpsg'],
    train_all_dataset['interact_mask'],
    train_all_dataset['interact_box'],
    train_all_dataset['grit_d_offline'],
    train_all_dataset['grit_cond_d_offline'],
    train_all_dataset['grit_r_offline'],
    train_all_dataset['grit_c_d_offline'],
    grand_cond_d,
    grand_cond_s,
    train_all_dataset['grand_d'],
    train_all_dataset['grand_s'],
    train_all_dataset['grand_c_d'],
    train_all_dataset['grand_c_s'],
]
for dataset in dataset_args:
    if dataset['type'] == 'SubSet':
        dataset['cfg'].setdefault('stage',2)
    else:
        dataset['stage'] = 2

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
# pretrained_pth = '/model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0905/iter_11500.pth'


model_dir = '/code/okapi-mllm/sketch_checkpoints/0828_iter37500'
projector = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=f"{model_dir}/projector",
    trust_remote_code=True,
)

vpt_encoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=f"{model_dir}/vpt_encoder",
    trust_remote_code=True,
)


model=dict(
    type=OkapiModel,
    # pretrained_pth=pretrained_pth,
    freeze_llm=True,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    freeze_projector=True,
    freeze_vpt_encoder=True,
    cutoff_len=cutoff_len,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        # pretrained_model_name_or_path=vicuna_7b_path,
        pretrained_model_name_or_path=model_dir,
        trust_remote_code=True),
    visual_encoder=clip_patch14_336['visual_encoder'],
    vpt_encoder=vpt_encoder,
    projector=projector,
    ref_adapter=dict(
        # Padding Method (packing=False): Max length denotes max corresponding token num in single 'phrase-unit-refs' tuple.
        # Packing Method(packing=True): Max length denotes max corresponding token num in single batch, 
        # and each token with a start and end token, like [ref_start]<REF>[ref_end]
        phrase_max_length=100,
        unit_max_length=50,
        ref_max_length=100,
        # phrase_max_length=1024,
        # unit_max_length=1024,
        # ref_max_length=300,
        # packing=True,
        packing=False,
        d_input=4096,
        d_model=1024,
        n_heads=8,
        dropout=0.1,
        d_ffn=2048,
        num_layers=3,
    ),
    visual_decoder=dict(
        box=dict(
            num_queries=100,
            quries_input_dim=1024,
            # quries_input_dim=4096,
            encoder_input_transform='resize_concat',
            encoder_input_index=[8, 16, 23], # clip-vit features
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
            num_queries=30,
            quries_input_dim=1024, # ref adapter
            # quries_input_dim=4096, # no ref adapter
            encoder_input_transform='multiple_select',
            encoder_input_index=[8, 16, 23], # clip-vit features, 23 equal to -2
            encoder_input_dim=[1024, 1024, 1024],
            #region query decoder config
            decoder_layers=6,
            decoder_ffn_dim=2048,
            decoder_attention_heads=8,
            decoder_layerdrop=0.0,
            pre_norm=False,
            activation_function="relu",
            d_model=256,
            dropout=0.1,
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
        box=0.5,
        mask=0.5
    ))
# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoModel
from mmengine.config import read_base

with read_base():
    from ._base_.models.all_visual_encoders import *
    from ._base_.datasets.vt_train_dataset_stage2 import *
    from ._base_.datasets.vt_val_dataset_stage2 import *
    from ._base_.models.vt_plug_vicuna_7b import *
    from ._base_.schedules.schedule import *
    from ._base_.default_runtime import *


# Data configs
batch_size = 15  # per_device
dataloader_num_workers = 4

# dataset grand det and seg
# dataset_args = [
#     train_all_dataset['res_refcoco'],
#     train_all_dataset['res_refcocoa'],
#     train_all_dataset['res_refcocog'],
    # train_all_dataset['llavag_gcg'],
    # train_all_dataset['openpsg'],
    # train_all_dataset['interact_mask'],
    # grand_cond_s,
    # train_all_dataset['grand_s'],
    # train_all_dataset['grand_c_s'],
# ]
for dataset in dataset_args:
    if dataset['type'] == 'SubSet':
        dataset['cfg'].setdefault('stage',2)
    else:
        dataset['stage'] = 2

train_dataset = dict(
    type=VTInstructDataset,
    dataset=dataset_args,
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
    sampler=dict(type=DefaultSampler, shuffle=True, seed=42),
    collate_fn=dict(type=vt_collate_fn))

val_cfg = None

# config models
# pretrained_pth = 'checkpoints/vicuna_7b/stage2/0828/iter_64500.pth'

model_dir = 'checkpoints/vicuna_7b/hf_model/0828_nodecoder_iter64500'


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

llm=dict(
    type=AutoModelForCausalLM.from_pretrained,
    # pretrained_model_name_or_path=vicuna_7b_path,
    pretrained_model_name_or_path=model_dir,
    trust_remote_code=True)

model=dict(
    type=VTPlugModel,
    # pretrained_pth=pretrained_pth,
    freeze_llm=False,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    llm=llm,
    visual_encoder=clip_patch14_336['visual_encoder'],
    visual_tower=clip_convnext_512['visual_encoder'],
    vpt_encoder=vpt_encoder,
    projector=projector,
    # llm=dict(
    #     type=AutoModelForCausalLM.from_pretrained,
    #     pretrained_model_name_or_path=vicuna_7b_path,
    #     trust_remote_code=True),
    # vpt_encoder=dict(
    #     strategy='pooling',
    #     patch_size=vpt_patch_size,
    #     num_patches = vpt_num_patches,
    #     visual_hidden_size=visual_hidden_size,
    #     use_mask_token=True,
    #     use_projector=False
    # ),
    visual_decoder=dict(
        box=dict(
            use_group_matcher=True,
            num_queries=100,
            # quries_input_dim=256,
            quries_input_dim=4096,
            encoder_input_transform='resize_concat',
            # encoder_input_dim shape = [[16, 16, 1024], [32, 32, 1024], [64, 64, 1024]]
            # encoder_input_index=[8, 16, 23], # clip-vit features
            # encoder_input_dim=[1024, 1024, 1024],
            # encoder_input_index=[0, 1, 2, 3], # clip-convnext features
            # encoder_input_dim=[192, 384, 768, 1536],

            encoder_input_index=[0, 1, 2, 4], # clip-convnext features with clip-vpt features
            encoder_input_dim=[192, 384, 768, 1024],

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
            use_group_matcher=True,
            num_queries=30,
            # quries_input_dim=256,
            quries_input_dim=4096,
            encoder_input_transform='multiple_select',
            # encoder_input_dim shape = [[16, 16, 1024], [32, 32, 1024], [64, 64, 1024]]
            # encoder_input_index=[8, 16, 23],   # [3, 2, 1], [-2,-2,-2]
            # encoder_input_dim=[1024, 1024, 1024],
            # encoder_input_index=[0, 1, 2, 3], # clip-convnext features
            # encoder_input_dim=[192, 384, 768, 1536],  

            encoder_input_index=[0, 1, 2, 4], # clip-convnext features with clip-vpt features
            encoder_input_dim=[192, 384, 768, 1024],
            
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
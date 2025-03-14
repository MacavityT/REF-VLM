# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoModel
from mmengine.config import read_base
with read_base():
    from ._base_.models.ref_vlm_encoders import clip_patch14_336, clip_convnext_320, clip_convnext_512
    from ._base_.datasets.vt_train_dataset_stage2 import *
    from ._base_.datasets.vt_val_dataset_stage2 import *
    from ._base_.models.ref_vlm_vicuna_7b import *
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
# for dataset in dataset_args:
#     if dataset['type'] == 'SubSet':
#         dataset['cfg'].setdefault('stage',2)
#     else:
#         dataset['stage'] = 2

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

model_dir = 'checkpoints/vicuna_7b/hf_model/0914_full_512_0124_iter68871'


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

box_decoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=f"{model_dir}/box_decoder",
    trust_remote_code=True,
)

mask_decoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=f"{model_dir}/mask_decoder",
    trust_remote_code=True,
)


model=dict(
    type=REFVLMModel,
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
    visual_decoder=dict(
        box=box_decoder,
        mask=mask_decoder,
    ),
    loss_coefficient=dict(
        llm=1,
        box=0.5,
        mask=0.5
    ))
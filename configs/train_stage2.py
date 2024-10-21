# Copyright (c) OpenMMLab. All rights reserved.
from transformers import AutoModel
from mmengine.config import read_base

with read_base():
    from ._base_.models.all_visual_encoders import clip_patch14_336
    from ._base_.datasets.vt_train_dataset_stage2 import *
    # from ._base_.datasets.sketch_train_dataset_stage2 import *
    from ._base_.datasets.vt_val_dataset_stage2 import *
    from ._base_.models.vt_plug_vicuna_7b import *
    from ._base_.schedules.schedule import *
    from ._base_.default_runtime import *

# Data configs
batch_size = 15  # per_device
dataloader_num_workers = 4


train_dataset = dict(
    type=VTInstructDataset,
    dataset=dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=dict(
        function=vt_map_fn_stage2,
        args = dict(
            # use_cot=False,   # use_cot
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
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=vt_collate_fn))

val_cfg = None

# config models
# pretrained_pth = 'checkpoints/vicuna_7b/stage1/0510_1_20_gc_rvg/iter_3558.pth'

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



model = dict(
    type=VTPlugModel,
    # pretrained_pth=pretrained_pth,
    freeze_llm=False,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    # llm=dict(
    #     type=AutoModelForCausalLM.from_pretrained,
    #     pretrained_model_name_or_path=vicuna_7b_path,
    #     trust_remote_code=True),
    llm=llm,
    visual_encoder=clip_patch14_336['visual_encoder'],
    vpt_encoder=vpt_encoder,
    projector=projector,
    # vpt_encoder=dict(
    #     strategy='pooling',
    #     patch_size=vpt_patch_size,
    #     num_patches = vpt_num_patches,
    #     visual_hidden_size=visual_hidden_size,
    #     use_mask_token=True,
    #     use_projector=False
    # ),
    loss_coefficient=dict(
        llm=1.,
        box=0.5,
        mask=0.5
    )
)
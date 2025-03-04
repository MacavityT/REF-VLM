# Copyright (c) OpenMMLab. All rights reserved.

from transformers import AutoModelForCausalLM
from vt_plug.model.external_modules.SAM import build_sam_plug, build_sam_preprocessor
from vt_plug.evaluation.metrics.single_metric import (
    ImgCapComputeMetrics,
    VQAComputeMetrics,
    COTComputeMetrics,
    LabelsComputeMetrics,
    PopeComputeMetrics,
    RECComputeMetrics,
    RESComputeMetrics,
    GCGComputeMetrics,
    DETComputeMetrics,
    SEGComputeMetrics,
)
from transformers import AutoModel
from mmengine.config import read_base
with read_base():
    from .._base_.models.all_tokenizers import *
    from .._base_.models.all_visual_encoders import *
    from .._base_.datasets.vt_test_dataset_stage2 import *
    from .._base_.datasets.vt_train_dataset_stage2 import *
    from .._base_.models.vt_plug_vicuna_7b import *
    # from ._base_.schedules.schedule import *
    from .._base_.default_runtime import *

# Data
test_cfg = dict(type='TestLoop')
dataloader_num_workers = 8
dataset_name = 'res_refcoco_val'
eval_type = 'caption'
prefix = 'reg_box'
chunk = 0

save_dir = 'work_dirs/ablation/0303_vpt_ours/eval'
model_dir = ''


if prefix == 'reg_box':
    test_evaluator = dict(
        type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        dict(
            type='SubSet',
            portion=1,
            # portion=1/3,
            do_shuffle=False,
            seed=43,
            # cfg=test_all_dataset['reg_refcocoa_unc_testa'],
            # cfg=test_all_dataset['interact_reg']
            cfg=test_all_dataset['reg_refcocog_umd_test_box'],
            )
    ]
elif prefix == 'reg_mask':
    test_evaluator = dict(
        type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        test_all_dataset['reg_refcocog_umd_test_mask'],
    ]
else:
    pass


test_dataset = dict(
    type=VTInstructDataset,
    dataset=test_dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=dict(
        function=vt_map_fn_stage2,
    ),
    template_map_fn=dict(
        type=vt_template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

test_dataloader = dict(
    batch_size=1,
    num_workers=dataloader_num_workers,
    dataset=test_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=vt_collate_fn))


if model_dir != '':
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
        pretrained_model_name_or_path=model_dir,
        trust_remote_code=True)
else:
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=vicuna_7b_path,
        trust_remote_code=True)
    vpt_encoder=dict(
        strategy='pooling',
        patch_size=vpt_patch_size,
        num_patches = vpt_num_patches,
        visual_hidden_size=visual_hidden_size,
        use_mask_token=True,
        use_projector=False,
        legacy=False
    )
    projector = None

model=dict(
    type=VTPlugModel,
    freeze_llm=True,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    llm=llm,
    visual_encoder=clip_patch14_336['visual_encoder'],
    vpt_encoder=vpt_encoder,
    projector=projector,
    loss_coefficient=dict(
        llm=1,
    ))
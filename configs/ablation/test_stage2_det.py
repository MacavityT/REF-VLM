# Copyright (c) OpenMMLab. All rights reserved.

from transformers import AutoModelForCausalLM
from ref_vlm.model.external_modules.SAM import build_sam_plug, build_sam_preprocessor
from ref_vlm.evaluation.metrics.single_metric import (
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
    from .._base_.models.ref_vlm_encoders import *
    from .._base_.datasets.vt_test_dataset_stage2 import *
    from .._base_.datasets.vt_train_dataset_stage2 import *
    from .._base_.models.ref_vlm_vicuna_7b import *
    # from ._base_.schedules.schedule import *
    from .._base_.default_runtime import *

# Data
test_cfg = dict(type='TestLoop')
dataloader_num_workers = 8
chunk = 0

save_dir = 'work_dirs/ablation/0305_det_no_match/eval_full'
model_dir = ''

test_evaluator = dict(
    type=DETComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, eval_type='class', prefix='coco_det')
test_dataset_args = [
    dict(
        type='SubSet',
        portion=1,
        do_shuffle=False,
        seed=43,
        cfg=test_all_dataset['coco_2017_box_val'],
        )
]

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

model=dict(
    type=REFVLMModel,
    freeze_llm=True,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    visual_encoder=clip_patch14_336['visual_encoder'],
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=vicuna_7b_path,
        trust_remote_code=True),
    projector=None,
    visual_decoder=dict(
        box=dict(
            use_group_matcher=False,
            num_queries=100,
            # quries_input_dim=256,
            quries_input_dim=4096,
            encoder_input_transform='resize_concat',
            # encoder_input_dim shape = [[16, 16, 1024], [32, 32, 1024], [64, 64, 1024]]
            encoder_input_index=[8, 16, 23], # clip-vit features
            encoder_input_dim=[1024, 1024, 1024],

            # encoder_input_index=[0, 1, 2, 3], # clip-convnext features
            # encoder_input_dim=[192, 384, 768, 1536],

            # encoder_input_index=[0, 1, 2, 4], # clip-convnext features with clip-vpt features
            # encoder_input_dim=[192, 384, 768, 1024],

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
    ),
    loss_coefficient=dict(
        llm=1,
    ))
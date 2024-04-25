# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.utils import PROMPT_TEMPLATE
from transformers import AutoModelForCausalLM
from xtuner.model import OkapiModel


from mmengine.config import read_base
with read_base():
    from ._base_.models.all_tokenizers import *
    from ._base_.models.all_visual_encoders import *
    from ._base_.datasets.okapi_test_dataset_stage1 import *
    # from ._base_.schedules.schedule import *
    from ._base_.default_runtime import *


# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''
evaluation_images = 'https://llava-vl.github.io/static/images/view.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture']


prompt_template = PROMPT_TEMPLATE.vicuna

model = dict(
    type=OkapiModel,
    freeze_llm=True,
    tokenizer=vicuna_7b_path_tokenizer,
    freeze_visual_encoder=True,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=vicuna_7b_path,
        trust_remote_code=True),
    visual_encoder=clip_patch14_336['visual_encoder'])


# test_dataset_args = [
#     dict(
#         type='SubSet',
#         portion=1/100,
#         do_shuffle=True,
#         seed=43,
#         enforce_online=True,
#         cfg=test_all_dataset['vqav2_val'],
#             )
    
# ]

test_dataset_args = [
    dict(
        type='SubSet',
        portion=1/20,
        do_shuffle=True,
        seed=43,
        enforce_online=True,
        cfg=test_all_dataset['caption'],
            )
    
]

okapi_dataset_test = dict(
    type=OkapiDataset,
    dataset=test_dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    tokenizer=vicuna_7b_path_tokenizer,
    dataset_map_fn=okapi_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

test_dataloader = dict(
    batch_size=1,
    num_workers=dataloader_num_workers,
    dataset=okapi_dataset_test,
    sampler=dict(type=DefaultSampler, shuffle=True),
    collate_fn=dict(type=default_collate_fn))


# test_evaluator = dict(
#     type=VQAComputeMetrics, tokenizer=vicuna_7b_path_tokenizer, prefix='vqa')

test_evaluator = dict(
    type=ImgCapComputeMetrics, tokenizer=vicuna_7b_path_tokenizer, prefix='caption')

# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook, tokenizer=vicuna_7b_path_tokenizer),
    dict(
        type=EvaluateChatHook,
        tokenizer=vicuna_7b_path_tokenizer,
        image_processor=clip_patch14_336['image_processor'],
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template)
]
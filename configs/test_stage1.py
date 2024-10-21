# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.utils import PROMPT_TEMPLATE
from transformers import AutoModelForCausalLM
from model import VTPlugModel

from mmengine.config import read_base
with read_base():
    from ._base_.models.all_tokenizers import *
    from ._base_.models.all_visual_encoders import *
    from ._base_.datasets.vt_test_dataset_stage1 import *
    from ._base_.models.vt_plug_vicuna_7b import *
    # from ._base_.schedules.schedule import *
    from ._base_.default_runtime import *

batch_size = 32  # per_device
dataloader_num_workers = 8

# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''
evaluation_images = 'https://llava-vl.github.io/static/images/view.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture']

save_dir = 'checkpoints/vicuna_7b/stage1/0510_1_20_gc_rvg/eval3558'


# test_dataset_args = [
#     dict(
#         type='SubSet',
#         portion=1/100,
#         do_shuffle=True,
#         seed=43,
#         cfg=test_all_dataset['vqav2_val'],
#             )
    
# ]

# test_dataset_args = [
#     dict(
#         type='SubSet',
#         portion=1/1000,
#         do_shuffle=True,
#         seed=43,
#         cfg=test_all_dataset['caption'],
#             )
    
# ]


# test_dataset_args = [
#     dict(
#         type='SubSet',
#         # portion=1/140,
#         portion=1/3,
#         do_shuffle=False,
#         seed=43,
#         cfg=test_all_dataset['reg_refcocoa_unc_testa'],
#         # cfg=test_all_dataset['interact_reg']
#         )
# ]


test_dataset_args = [
    dict(
        type='SubSet',
        portion=1/5,
        do_shuffle=False,
        seed=43,
        cfg=test_all_dataset['reg'],
        )
]

dataset_test = dict(
    type=VTInstructDataset,
    pretokenize=False,
    dataset=test_dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=vt_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

test_dataloader = dict(
    batch_size=1,
    num_workers=dataloader_num_workers,
    dataset=dataset_test,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=default_collate_fn))


# test_evaluator = dict(
#     type=VQAComputeMetrics, tokenizer=tokenizer, prefix='vqa')

test_evaluator = dict(
    type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=1, save_dir=save_dir, prefix='reg')


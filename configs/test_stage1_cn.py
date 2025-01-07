# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.utils import PROMPT_TEMPLATE
from transformers import AutoModelForCausalLM
from vt_plug.model import VTPlugModel

from mmengine.config import read_base
with read_base():
    from ._base_.models.all_tokenizers import *
    from ._base_.models.all_visual_encoders import *
    from ._base_.datasets.vt_test_dataset_stage1 import *
    from ._base_.models.vt_plug_qwen import *
    # from ._base_.schedules.schedule import *
    from ._base_.default_runtime import *

dataloader_num_workers = 8


save_dir = 'checkpoints/Qwen2.5/xtuner_output/stage1/1231/eval'


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
        cfg=test_all_dataset['caption'],
        )
]

dataset_test = dict(
    type=VTInstructDataset,
    dataset=test_dataset_args,
    # image_processor=clip_patch14_336['image_processor'],
    image_processor=clip_big14_224['image_processor'],
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
    type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=1, save_dir=save_dir, prefix='caption')


# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.utils import PROMPT_TEMPLATE

from mmengine.config import read_base
with read_base():
    from _base_.models.all_tokenizers import vicuna_7b_path, vicuna_7b_path_tokenizer
    from _base_.models.all_visual_encoders import clip_patch14_336
    from _base_.datasets.okapi_train_stage1 import *
    from _base_.schedules.schedule import *
    from _base_.runtimes.okapi_vicuna_runtime import *
    from _base_.runtimes.default_runtime import *


# Evaluate the generation performance during the training
evaluation_freq = 500
SYSTEM = ''
evaluation_images = 'https://llava-vl.github.io/static/images/view.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture']

prompt_template = PROMPT_TEMPLATE.vicuna

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
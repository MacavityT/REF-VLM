# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.utils import PROMPT_TEMPLATE
from transformers import AutoModelForCausalLM
from xtuner.model import OkapiModel
from xtuner.evaluation.metrics.single_metric import ImgCapComputeMetrics,VQAComputeMetrics,COTComputeMetrics,LabelsComputeMetrics
from xtuner.dataset.map_fns import (
    okapi_map_fn_stage2,
    okapi_template_map_fn_factory
)
from xtuner.dataset.collate_fns import okapi_collate_fn
from transformers import AutoModel
from mmengine.config import read_base
with read_base():
    from ._base_.models.all_tokenizers import *
    from ._base_.models.all_visual_encoders import *
    from ._base_.datasets.okapi_test_dataset_stage2 import *
    from ._base_.models.okapi_vicuna_7b import *
    # from ._base_.schedules.schedule import *
    from ._base_.default_runtime import *

# Data
prompt_template = PROMPT_TEMPLATE.okapi
max_length = 10000  # use cutoff lens instead
cutoff_len = 4096
dataloader_num_workers = 8
vrt_length = 64
ref_length = 1

eval_type = 'phrase'
prefix = 'reg'

save_dir = '/model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0702/eval3012'

if prefix == 'vqa':
    test_evaluator = dict(
        type=VQAComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix='vqa')
    test_dataset_args = [
    dict(
        type='SubSet',
        portion=1/200,
        do_shuffle=False,
        seed=43,
        enforce_online=True,
        cfg=test_all_dataset['vqav2_val'],
        )
]
elif prefix == 'caption':
    test_evaluator = dict(
        type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix='caption')
    test_dataset_args = [
        dict(
            type='SubSet',
            portion=1/20,
            do_shuffle=False,
            seed=43,
            enforce_online=True,
            cfg=test_all_dataset['caption'],
            )
    ]

elif prefix == 'reg':
    test_evaluator = dict(
        type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix='reg')
    test_dataset_args = [
        dict(
            type='SubSet',
            portion=1/140,
            # portion=1/3,
            do_shuffle=False,
            seed=43,
            enforce_online=True,
            # cfg=test_all_dataset['reg_refcocoa_unc_testa'],
            cfg=test_all_dataset['interact_reg']
            )
    ]

elif (prefix == 'cot') or (prefix == 'vrt') or (prefix == 'cot_vrt'):
    test_evaluator = dict(
        type=COTComputeMetrics, tokenizer=tokenizer, stage=2, eval_type=eval_type, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        dict(
            type='SubSet',
            portion=1,
            do_shuffle=False,
            seed=43,
            enforce_online=True,
            cfg=test_all_dataset['rec_refcocoa_unc_testa'],
            )
    ]

elif prefix == 'label':
    assert eval_type == 'phrase' or eval_type == 'count'
    test_evaluator = dict(
        type=LabelsComputeMetrics, tokenizer=tokenizer, stage=2, eval_type=eval_type, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        dict(
            type='SubSet',
            portion=1/1000,
            do_shuffle=False,
            seed=43,
            enforce_online=True,
            cfg=test_all_dataset['grand_d_s'],
            )
    ]


test_dataset = dict(
    type=OkapiDataset,
    pretokenize=False,
    dataset=test_dataset_args,
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

test_dataloader = dict(
    batch_size=1,
    num_workers=dataloader_num_workers,
    dataset=test_dataset,
    sampler=dict(type=DefaultSampler, shuffle=False),
    collate_fn=dict(type=okapi_collate_fn))

model = dict(
    type=OkapiModel,
    freeze_llm=True,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=vicuna_7b_path,
        trust_remote_code=True),
    visual_encoder=clip_patch14_336['visual_encoder'])

# Copyright (c) OpenMMLab. All rights reserved.
from xtuner.engine.hooks import DatasetInfoHook, EvaluateChatHook
from xtuner.utils import PROMPT_TEMPLATE
from transformers import AutoModelForCausalLM
from xtuner.model import OkapiModel
from xtuner.evaluation.metrics.single_metric import (
    ImgCapComputeMetrics,
    VQAComputeMetrics,
    COTComputeMetrics,
    LabelsComputeMetrics,
    PopeComputeMetrics,
)
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
cutoff_len = 4096  # 4096
dataloader_num_workers = 8
visual_hidden_size = 1024
vrt_length = 256
ref_length = 1
vpt_num_patches = 9
vpt_patch_size = 8 # sqrt(576/9)=8
cot_weight = 1
vrt_weight = 1

eval_type = 'vqa'
prefix = 'vqa_box_pron'
chunk = 8

save_dir = '/model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0718/eval59525'

if prefix == 'vqa':
    test_evaluator = dict(
        type=VQAComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix='vqav2',chunk=chunk)
    test_dataset_args = [
        # dict(
        #     type='SubSet',
        #     portion=1/200,
        #     do_shuffle=False,
        #     seed=43,
        #     cfg=test_all_dataset['vqav2_val'],
        #     )
        test_all_dataset['vqav2_test_8'],
    ]

elif prefix == 'vqa_point_pron':
    test_evaluator = dict(
        type=VQAComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        test_all_dataset['point_twice_oq_p_test'],
    ]

elif prefix == 'vqa_point_sup':
    test_evaluator = dict(
        type=VQAComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        test_all_dataset['point_twice_sq_p_test'],
    ]

elif prefix == 'vqa_point_gen':
    test_evaluator = dict(
        type=VQAComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        test_all_dataset['point_twice_gq_p_test'],
    ]

elif prefix == 'vqa_box_pron':
    test_evaluator = dict(
        type=VQAComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        test_all_dataset['point_twice_oq_b_test'],
    ]

elif prefix == 'vqa_box_sup':
    test_evaluator = dict(
        type=VQAComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        test_all_dataset['point_twice_sq_b_test'],
    ]

elif prefix == 'vqa_box_gen':
    test_evaluator = dict(
        type=VQAComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        test_all_dataset['point_twice_gq_b_test'],
    ]


elif prefix == 'caption_coco':
    test_evaluator = dict(
        type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix='caption')
    test_dataset_args = [
        dict(
            type='SubSet',
            portion=1/20,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset['caption'],
            )
    ]

elif prefix == 'caption_flickr':
    test_evaluator = dict(
    type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix='caption')
    test_dataset_args = [
        test_all_dataset['flickr_test_without_box'],
    ]

elif prefix == 'pope_random':
    test_evaluator = dict(
    type=PopeComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix='pope_random')
    test_dataset_args = [
        test_all_dataset['coco_pope_random_q_a'],
    ]

elif prefix == 'pope_popular':
    test_evaluator = dict(
    type=PopeComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix='pope_popular')
    test_dataset_args = [
        test_all_dataset['coco_pope_popular_q_a'],
    ]

elif prefix == 'pope_adversarial':
    test_evaluator = dict(
    type=PopeComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix='pope_adversarial')
    test_dataset_args = [
        test_all_dataset['coco_pope_adversarial_q_a'],
    ]

elif prefix == 'reg':
    test_evaluator = dict(
        type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix='reg')
    test_dataset_args = [
        dict(
            type='SubSet',
            portion=1,
            # portion=1/3,
            do_shuffle=False,
            seed=43,
            # cfg=test_all_dataset['reg_refcocoa_unc_testa'],
            # cfg=test_all_dataset['interact_reg']
            cfg=test_all_dataset['reg_refcocog_umd_test'],
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
            cfg=test_all_dataset['grand_d_s'],
            )
    ]


test_dataset = dict(
    type=OkapiDataset,
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
    visual_encoder=clip_patch14_336['visual_encoder'],
    vpt_encoder=dict(
        strategy='pooling',
        patch_size=vpt_patch_size,
        num_patches = vpt_num_patches,
        visual_hidden_size=visual_hidden_size,
        use_mask_token=False,
        use_projector=False,
    ),
    visual_sync_tuner=dict(
        num_layers=3,
        num_queries=vrt_length,
        d_input=4096,
        d_model=512,
        d_ffn=2048,
        output_dim=3,
        num_heads=8,
        dropout=0.1,
        ratio=0.5
    ),
    cot_weight=cot_weight,
    vrt_weight=vrt_weight)

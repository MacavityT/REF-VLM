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
    RECComputeMetrics,
    RESComputeMetrics,
    GCGComputeMetrics,
    DETComputeMetrics,
    SEGComputeMetrics,
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
vrt_length = 0
ref_length = 1
vpt_num_patches = 9
vpt_patch_size = 8 # sqrt(576/9)=8
ref_box_num = 100
ref_mask_num = 30
ref_box_queries = ref_box_num * ref_length
ref_mask_queries = ref_mask_num * ref_length

dataset_name = 'res_refcocog_test'
eval_type = 'reg'
prefix = 'res'
chunk = 8

save_dir = '/model/Aaronzhu/OkapiModel/vicuna_7b/stage2/1001_mask_512_nomatcher/eval3355'

if prefix == 'okvqa':
    test_evaluator = dict( 
        type=VQAComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        test_all_dataset['okvqa'],
    ]

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
        type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
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
    type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
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

elif prefix == 'reg_box':
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


elif prefix == 'rec':
    test_evaluator = dict(
        type=RECComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix,dataset_name=dataset_name)
    test_dataset_args = [
        test_all_dataset[f'{dataset_name}'],
        # dict(
        #     type='SubSet',
        #     portion=1/10,
        #     do_shuffle=False,
        #     seed=43,
        #     cfg=test_all_dataset[f'{dataset_name}'],
        #     )
    ]

elif prefix == 'res':
    test_evaluator = dict(
        type=RESComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix,dataset_name=dataset_name)
    test_dataset_args = [
        # test_all_dataset[f'{dataset_name}'],
        dict(
            type='SubSet',
            portion=1/10,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset[f'{dataset_name}'],
            )
    ]

elif prefix == 'interactive_res':
    test_evaluator = dict(
        type=RESComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix,dataset_name=dataset_name)
    test_dataset_args = [
        # test_all_dataset[f'{dataset_name}'],
        dict(
            type='SubSet',
            portion=1/10,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset[f'{dataset_name}'],
            )
    ]

elif prefix == 'gcg_box':
    test_evaluator = dict(
        type=GCGComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, eval_type='whole', mask=False, prefix=prefix)
    test_dataset_args = [
        # test_all_dataset['rec_refcocog_umd_test'],
        dict(
            type='SubSet',
            portion=1/10,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset['flickr_eval_with_box'],
            )
    ]

elif prefix == 'gcg_mask':
    test_evaluator = dict(
        type=GCGComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, eval_type='whole', mask=True, prefix=prefix)
    test_dataset_args = [
        dict(
            type='SubSet',
            portion=2/3,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset['cocogcg_val'],
            )
    ]

elif prefix == 'gcg_mask_test':
    test_evaluator = dict(
        type=GCGComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, eval_type='whole', mask=True, prefix=prefix)
    test_dataset_args = [
        dict(
            type='SubSet',
            portion=1/2,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset['cocogcg_test'],
            )
    ]

elif prefix == 'coco_det':
    test_evaluator = dict(
        type=DETComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, eval_type='class', prefix=prefix)
    test_dataset_args = [
        # test_all_dataset['rec_refcocog_umd_test'],
        dict(
            type='SubSet',
            portion=1,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset['coco_2017_box_val'],
            )
    ]

elif prefix == 'lvis_det':
    test_evaluator = dict(
        type=DETComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, eval_type='class', prefix=prefix)
    test_dataset_args = [
        # test_all_dataset['rec_refcocog_umd_test'],
        dict(
            type='SubSet',
            portion=1,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset['lvis_box_test'],
            )
    ]

elif prefix == 'cityscapes_instance':
    test_evaluator = dict(
        type=SEGComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, eval_type='class', prefix=prefix)
    test_dataset_args = [
        # test_all_dataset['rec_refcocog_umd_test'],
        dict(
            type='SubSet',
            portion=1,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset['test_cityscapes_instance'],
            )
    ]

elif prefix == 'ade20k_instance':
    test_evaluator = dict(
        type=SEGComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, eval_type='class', prefix=prefix)
    test_dataset_args = [
        # test_all_dataset['rec_refcocog_umd_test'],
        dict(
            type='SubSet',
            portion=1,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset['test_ade20_with_instance'],
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
    image_tower_processor=clip_convnext_512['image_processor'],
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


model=dict(
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
    visual_tower=clip_convnext_512['visual_encoder'],
    # visual_tower=clip_convnext_320['visual_encoder'],
    vpt_encoder=dict(
        strategy='pooling',
        patch_size=vpt_patch_size,
        num_patches = vpt_num_patches,
        visual_hidden_size=visual_hidden_size,
        use_mask_token=True,
        use_projector=False
    ),
    visual_decoder=dict(
        box=dict(
            use_group_matcher=True,
            num_queries=ref_box_queries,
            # quries_input_dim=256,
            quries_input_dim=4096,
            encoder_input_transform='resize_concat',
            # encoder_input_dim shape = [[16, 16, 1024], [32, 32, 1024], [64, 64, 1024]]
            # encoder_input_index=[8, 16, 23], # clip-vit features
            # encoder_input_dim=[1024, 1024, 1024],
            # encoder_input_index=[0, 1, 2, 3], # clip-convnext features
            # encoder_input_dim=[192, 384, 768, 1536],

            encoder_input_index=[0, 1, 2, 4], # clip-convnext features with clip-vpt features
            encoder_input_dim=[192, 384, 768, 1024],

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
        mask=dict(
            use_group_matcher=True,
            num_queries=ref_mask_queries,
            # quries_input_dim=256,
            quries_input_dim=4096,
            encoder_input_transform='multiple_select',
            # encoder_input_dim shape = [[16, 16, 1024], [32, 32, 1024], [64, 64, 1024]]
            # encoder_input_index=[8, 16, 23],   # [3, 2, 1], [-2,-2,-2]
            # encoder_input_dim=[1024, 1024, 1024],
            # encoder_input_index=[0, 1, 2, 3], # clip-convnext features
            # encoder_input_dim=[192, 384, 768, 1536],  

            encoder_input_index=[0, 1, 2, 4], # clip-convnext features with clip-vpt features
            encoder_input_dim=[192, 384, 768, 1024],
            
            #region query decoder config
            decoder_layers=6,
            decoder_ffn_dim=2048,
            decoder_attention_heads=8,
            decoder_layerdrop=0.0,
            pre_norm=False,
            activation_function="relu",
            d_model=256,
            dropout=0.1,
            attention_dropout=0.0,
            activation_dropout=0.0,
            #endregion
            #region pixel decoder config
            encoder_layers=6, 
            fpn_feature_size=256,
            mask_feature_size=256,
            feature_strides=[4, 8, 16, 32],
            common_stride=4,
            encoder_feedforward_dim=1024,
            mask_loss_coefficient=20,
            dice_loss_coefficient=1,
            #endregion
        ),
    ),
    loss_coefficient=dict(
        llm=1,
        box=0.5,
        mask=0.5
    ))
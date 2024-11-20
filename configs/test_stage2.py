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
    from ._base_.models.all_tokenizers import *
    from ._base_.models.all_visual_encoders import *
    from ._base_.datasets.vt_test_dataset_stage2 import *
    from ._base_.models.vt_plug_vicuna_7b import *
    # from ._base_.schedules.schedule import *
    from ._base_.default_runtime import *

# Data
test_cfg = dict(type='TestLoop')
dataloader_num_workers = 8
dataset_name = 'res_refcoco_testa'
eval_type = 'caption'
prefix = 'res'
chunk = 0

save_dir = 'checkpoints/vicuna_7b/finetune/1119_sam_res/eval4000'
# model_dir = 'checkpoints/vicuna_7b/hf_model/0914_nodecoder_iter11500'
model_dir = ''

sam_preprocessor = dict(
    type=build_sam_preprocessor,
    target_length=1024
)




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
        dict(
            type='SubSet',
            portion=1/200,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset['vqav2_val'],
            )
        # test_all_dataset['vqav2_test_0'],
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

elif prefix == 'caption_nocaps':
    test_evaluator = dict(
        type=ImgCapComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        dict(
            type='SubSet',
            portion=1/3,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset['nocaps_val'],
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
    type=VTInstructDataset,
    dataset=test_dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    # image_tower_processor=clip_convnext_512['image_processor'],
    image_tower_processor=sam_preprocessor,
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
        use_projector=False
    )
    projector = None



# model=dict(
#     type=VTPlugModel,
#     freeze_llm=True,
#     tokenizer=tokenizer,
#     freeze_visual_encoder=True,
#     cutoff_len=cutoff_len,
#     llm=llm,
#     visual_encoder=clip_patch14_336['visual_encoder'],
#     visual_tower=clip_convnext_512['visual_encoder'],
#     # visual_tower=clip_convnext_320['visual_encoder'],
#     vpt_encoder=vpt_encoder,
#     projector=projector,
#     # ref_adapter=dict(
#     #     # Padding Method (packing=False): Max length denotes max corresponding token num in single 'phrase-unit-refs' tuple.
#     #     # Packing Method(packing=True): Max length denotes max corresponding token num in single batch, 
#     #     # and each token with a start and end token, like [ref_start]<REF>[ref_end]

#     #     # packing=True,
#     #     # phrase_max_length=100,
#     #     # unit_max_length=50,
#     #     # ref_max_length=100,

#     #     # phrase_max_length=1024,
#     #     # unit_max_length=1024,
#     #     # ref_max_length=300,

#     #     mode='encode',
#     #     modality='visual',
#     #     max_position_embedding=4096,
#     #     d_input=4096,
#     #     d_model=4096,
#     #     n_heads=8,
#     #     dropout=0.1,
#     #     d_ffn=8192,
#     #     num_layers=3,
#     # ),
#     visual_decoder=dict(
#         box=dict(
#             use_group_matcher=True,
#             num_queries=100,
#             # quries_input_dim=256,
#             quries_input_dim=4096,
#             encoder_input_transform='resize_concat',
#             # encoder_input_dim shape = [[16, 16, 1024], [32, 32, 1024], [64, 64, 1024]]
#             # encoder_input_index=[8, 16, 23], # clip-vit features
#             # encoder_input_dim=[1024, 1024, 1024],
#             # encoder_input_index=[0, 1, 2, 3], # clip-convnext features
#             # encoder_input_dim=[192, 384, 768, 1536],

#             encoder_input_index=[0, 1, 2, 4], # clip-convnext features with clip-vpt features
#             encoder_input_dim=[192, 384, 768, 1024],

#             decoder_layers=6,
#             decoder_ffn_dim=2048,
#             decoder_attention_heads=8,
#             decoder_layerdrop=0.0,
#             activation_function="relu",
#             d_model=256,
#             dropout=0.1,
#             attention_dropout=0.0,
#             activation_dropout=0.0,
#             bbox_loss_coefficient=5,
#             giou_loss_coefficient=2,
#         ),
#         mask=dict(
#             use_group_matcher=True,
#             num_queries=30,
#             # quries_input_dim=256,
#             quries_input_dim=4096,
#             encoder_input_transform='multiple_select',
#             # encoder_input_dim shape = [[16, 16, 1024], [32, 32, 1024], [64, 64, 1024]]
#             # encoder_input_index=[8, 16, 23],   # [3, 2, 1], [-2,-2,-2]
#             # encoder_input_dim=[1024, 1024, 1024],
#             # encoder_input_index=[0, 1, 2, 3], # clip-convnext features
#             # encoder_input_dim=[192, 384, 768, 1536],  

#             encoder_input_index=[0, 1, 2, 4], # clip-convnext features with clip-vpt features
#             encoder_input_dim=[192, 384, 768, 1024],
            
#             #region query decoder config
#             decoder_layers=6,
#             decoder_ffn_dim=2048,
#             decoder_attention_heads=8,
#             decoder_layerdrop=0.0,
#             pre_norm=False,
#             activation_function="relu",
#             d_model=256,
#             dropout=0.1,
#             attention_dropout=0.0,
#             activation_dropout=0.0,
#             #endregion
#             #region pixel decoder config
#             encoder_layers=6, 
#             fpn_feature_size=256,
#             mask_feature_size=256,
#             feature_strides=[4, 8, 16, 32],
#             common_stride=4,
#             encoder_feedforward_dim=1024,
#             mask_loss_coefficient=20,
#             dice_loss_coefficient=1,
#             #endregion
#         ),
#     ),
#     loss_coefficient=dict(
#         llm=1,
#         box=1,
#         mask=1
#     ))



model=dict(
    type=VTPlugModel,
    # pretrained_pth=pretrained_pth,
    freeze_llm=False,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    llm=llm,
    visual_encoder=clip_patch14_336['visual_encoder'],
    visual_tower=None,
    vpt_encoder=vpt_encoder,
    projector=projector,
    visual_decoder=dict(
        mask=dict(
            type=build_sam_plug,
            version='huge',
            checkpoint='./checkpoints/SAM/sam_vit_h_4b8939.pth',
            freeze_mask_decoder=False,
            num_queries=100,
            quries_input_dim=4096,
            d_model=256,
            mask_loss_coefficient=20,
            dice_loss_coefficient=1,
            use_group_matcher=True
        ),
    ),
    loss_coefficient=dict(
        llm=1,
        mask=1
    ))
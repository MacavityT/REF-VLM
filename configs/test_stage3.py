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
    POSEComputeMetrics
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
dataloader_num_workers = 0

prefix = 'pose'

save_dir = 'checkpoints/vicuna_7b/stage2/1016_keypoint/eval34160'
model_dir = ''


if prefix == 'pose':
    test_evaluator = dict(
        type=POSEComputeMetrics, tokenizer=tokenizer, stage=2, save_dir=save_dir, prefix=prefix)
    test_dataset_args = [
        # test_all_dataset[f'{dataset_name}'],
        dict(
            type='SubSet',
            portion=1,
            do_shuffle=False,
            seed=43,
            cfg=test_all_dataset[f'keypoints2017_det'],
            )
    ]


test_dataset = dict(
    type=VTInstructDataset,
    dataset=test_dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    image_tower_processor=clip_convnext_512['image_processor'],
    # image_tower_processor=sam_preprocessor,
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

    llm = dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=model_dir,
        trust_remote_code=True)
    
    pose_decoder = dict(
        type=AutoModel.from_pretrained,
        pretrained_model_name_or_path=f"{model_dir}/pose_decoder",
        trust_remote_code=True,
    )
    

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
    pose_decoder=dict(
            num_queries=100,
            encoder_input_transform='resize_concat',
            encoder_input_index=[0, 1, 2, 4], # clip-convnext features with clip-vpt features
            encoder_input_dim=[192, 384, 768, 1024],
            use_group_matcher=True,  # True
            use_auxiliary_loss=True,
            aux_loss_coefficient=0.5,
            box_config=dict(
                quries_input_dim=4096,
                decoder_layers=6,
                decoder_ffn_dim=2048,
                decoder_attention_heads=8,
                decoder_layerdrop=0.0,
                activation_function="relu",
                d_model=256,
                dropout=0.1,
                attention_dropout=0.0,
                activation_dropout=0.0,
                bbox_loss_coefficient=5, # 5
                giou_loss_coefficient=2, # 2
            ),
            keypoint_config=dict(
                quries_input_dim=256,
                decoder_layers=6,
                decoder_ffn_dim=2048,
                decoder_attention_heads=8,
                decoder_layerdrop=0.0,
                activation_function="relu",
                d_model=256,
                dropout=0.1,
                attention_dropout=0.0,
                activation_dropout=0.0,
                num_body_points=17,
                keypoint_loss_coefficient=2,  #2
                oks_loss_coefficient=2,  #2
                cls_loss_coefficient=1,  #1
            )
        )



model=dict(
    type=VTPlugModel,
    freeze_llm=True,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    llm=llm,
    visual_encoder=clip_patch14_336['visual_encoder'],
    visual_tower=clip_convnext_512['visual_encoder'],
    vpt_encoder=vpt_encoder,
    visual_decoder=dict(
        pose=pose_decoder,
    ),

    loss_coefficient=dict(
        llm=1,
        box=1,
        mask=1
    ))

import os
from transformers import AutoModel
from peft import PeftModel
from ref_vlm.model.external_modules.SAM import build_sam_plug, build_sam_preprocessor
from mmengine.config import read_base
with read_base():
    from ._base_.models.ref_vlm_encoders import clip_patch14_336,clip_convnext_320,clip_convnext_512
    from ._base_.datasets.vt_train_dataset_stage2 import *
    from ._base_.datasets.vt_val_dataset_stage2 import *
    from ._base_.models.ref_vlm_vicuna_7b import *

pretrained_pth = "checkpoints/vicuna_7b/finetune/1121_sam_rem/iter_500.pth"

sam_preprocessor = dict(
    type=build_sam_preprocessor,
    target_length=1024
)

infer_dataset = dict(
    type=VTInstructDataset,
    dataset=dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    image_tower_processor=sam_preprocessor,
    tokenizer=tokenizer,
    dataset_map_fn=dict(
        function=vt_map_fn_stage2,
    ),
    template_map_fn=dict(
        type=vt_template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    mode='inference')


model_dir = 'checkpoints/vicuna_7b/hf_model/0828_nodecoder_iter64500'

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
    # pretrained_model_name_or_path=vicuna_7b_path,
    pretrained_model_name_or_path=model_dir,
    trust_remote_code=True)

model=dict(
    type=REFVLMModel,
    pretrained_pth=pretrained_pth,
    freeze_llm=True,
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
            freeze_mask_decoder=True,
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
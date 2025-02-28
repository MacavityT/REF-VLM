import os
from transformers import AutoModel
from peft import PeftModel

from mmengine.config import read_base
with read_base():
    from ._base_.models.all_visual_encoders import clip_patch14_336,clip_convnext_320, clip_convnext_512
    from ._base_.datasets.vt_train_dataset_stage2 import *
    from ._base_.datasets.vt_val_dataset_stage2 import *
    from ._base_.models.vt_plug_vicuna_7b import *

# freezed_llm_dir = 'checkpoints/vicuna_7b/hf_model/0914_nodecoder_iter11500'
model_dir = "checkpoints/vicuna_7b/hf_model/0222_det_freeze_llm8000"
freezed_llm_dir = model_dir

projector = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=os.path.join(model_dir,'projector'),
    trust_remote_code=True,
)

vpt_encoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=os.path.join(model_dir,'vpt_encoder'),
    trust_remote_code=True,
)

vd_adapter = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=os.path.join(model_dir,'vd_adapter'),
    trust_remote_code=True,
)

box_decoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=os.path.join(model_dir,'box_decoder'),
    trust_remote_code=True,
)

mask_decoder = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=os.path.join(model_dir,'mask_decoder'),
    trust_remote_code=True,
)



infer_dataset = dict(
    type=VTInstructDataset,
    dataset=dataset_args,
    image_processor=clip_patch14_336['image_processor'],
    image_tower_processor=clip_convnext_512['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=dict(
        function=vt_map_fn_stage2,
    ),
    template_map_fn=dict(
        type=vt_template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    mode='inference')


model = dict(
    type=VTPlugModel,
    freeze_llm=True,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=cutoff_len,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=freezed_llm_dir,
        trust_remote_code=True),
    visual_encoder=clip_patch14_336['visual_encoder'],
    visual_tower=clip_convnext_512['visual_encoder'],
    projector=projector,
    vpt_encoder=vpt_encoder,
    # vd_adapter=vd_adapter,
    visual_decoder=dict(
        box=box_decoder,
        # mask=mask_decoder
    )
)
import os
from transformers import AutoModel
from peft import PeftModel

from mmengine.config import read_base
from vt_plug.dataset.map_fns import (
    vt_template_map_fn_factory
)
with read_base():
    from ._base_.models.all_visual_encoders import clip_patch14_336,clip_convnext_320,clip_convnext_512,clip_big14_224
    from ._base_.datasets.vt_train_dataset_stage1_cn import *
    from ._base_.datasets.vt_val_dataset_stage1 import *
    from ._base_.models.vt_plug_qwen import *


model_dir = "/code/VT-PLUG/checkpoints/Qwen2.5/hf_output/0102_iter30000"
max_length = int(2048 - (336 / 14)**2)

projector = dict(
    type=AutoModel.from_pretrained,
    pretrained_model_name_or_path=os.path.join(model_dir,'projector'),
    trust_remote_code=True,
)


infer_dataset = dict(
    type=VTInstructDataset,
    dataset=dataset_args,
    image_processor=clip_big14_224['image_processor'],
    tokenizer=tokenizer,
    dataset_map_fn=vt_map_fn,
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
    cutoff_len=2048,
    stage=1,
    projector=projector,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=model_dir,
        trust_remote_code=True),
    visual_encoder=clip_big14_224['visual_encoder'],
)

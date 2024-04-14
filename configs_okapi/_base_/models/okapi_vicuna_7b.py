_base_ = [
    'tokenizers.py',
    'visual_encoders.py'
]


from transformers import AutoModelForCausalLM, CLIPVisionModel
from xtuner.model import OkapiModel


llm_name_or_path = _base_.vicuna_7b_path
visual_encoder_name_or_path = _base_.clip_patch14_336_path

model = dict(
    type=OkapiModel,
    freeze_llm=True,
    freeze_visual_encoder=True,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True),
    visual_encoder=dict(
        type=CLIPVisionModel.from_pretrained,
        pretrained_model_name_or_path=visual_encoder_name_or_path))
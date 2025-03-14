from transformers import AutoModelForCausalLM
from ref_vlm.model import REFVLMModel
from mmengine.config import read_base
with read_base():
    from .all_tokenizers import vicuna_7b_path, vicuna_7b_path_tokenizer
    from .ref_vlm_encoders import clip_patch14_336

tokenizer = vicuna_7b_path_tokenizer
model = dict(
    type=REFVLMModel,
    freeze_llm=True,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=2048,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=vicuna_7b_path,
        trust_remote_code=True),
    visual_encoder=clip_patch14_336['visual_encoder'],
    loss_coefficient=dict(
        llm=1.)
)

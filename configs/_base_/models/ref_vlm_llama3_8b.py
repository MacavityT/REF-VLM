from transformers import AutoModelForCausalLM
from ref_vlm.model import REFVLMModel
from xtuner.utils import PROMPT_TEMPLATE, SYSTEM_TEMPLATE
from mmengine.config import read_base
with read_base():
    from .all_tokenizers import llama3_8b_path, llama3_8b_path_tokenizer
    from .ref_vlm_encoders import clip_patch14_336


prompt_template = PROMPT_TEMPLATE.llama3_chat
tokenizer = llama3_8b_path_tokenizer
model = dict(
    type=REFVLMModel,
    freeze_llm=True,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=4096,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llama3_8b_path,
        trust_remote_code=True),
    visual_encoder=clip_patch14_336['visual_encoder'])

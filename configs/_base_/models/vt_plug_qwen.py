from transformers import AutoModelForCausalLM,AutoModel
from vt_plug.model import VTPlugModel
from vt_plug.utils import PROMPT_TEMPLATE
from mmengine.config import read_base
with read_base():
    from .all_tokenizers import qwen2_7b_path, qwen2_7b_path_tokenizer
    from .all_visual_encoders import clip_patch14_336,clip_big14_224


prompt_template = PROMPT_TEMPLATE.qwen_chat
tokenizer = qwen2_7b_path_tokenizer
model = dict(
    type=VTPlugModel,
    freeze_llm=True,
    tokenizer=tokenizer,
    freeze_visual_encoder=True,
    cutoff_len=2048,
    stage=1,
    projector=dict(
        llm_hidden_size=3584,
        depth=2,
    ),
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=qwen2_7b_path,
        trust_remote_code=True),
    # visual_encoder=clip_patch14_336['visual_encoder'],
    visual_encoder=clip_big14_224['visual_encoder'],
    loss_coefficient=dict(llm=1.),
    )

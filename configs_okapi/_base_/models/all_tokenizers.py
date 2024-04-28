from transformers import AutoTokenizer

# lmsys/vicuna-7b-v1.5
vicuna_7b_path = '/model/Aaronzhu/Vicuna/7b_v1.5'

vicuna_7b_path_tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=vicuna_7b_path,
    trust_remote_code=True,
    padding_side='right')
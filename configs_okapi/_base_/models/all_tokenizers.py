from transformers import AutoTokenizer


vicuna_7b_path = '/model/Aaronzhu/Mistral-7b'

vicuna_7b_path_tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=vicuna_7b_path,
    trust_remote_code=True,
    padding_side='right')
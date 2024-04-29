from transformers import AutoTokenizer

# Vicuna 7b
vicuna_7b_path = '/model/Aaronzhu/Vicuna/7b_v1.5'
vicuna_7b_path_tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=vicuna_7b_path,
    trust_remote_code=True,
    padding_side='right')


# Mistral 7b
mistral_7b_path = '/model/Aaronzhu/Mistral-7b'
mistral_7b_path_tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=mistral_7b_path,
    trust_remote_code=True,
    padding_side='right')


# Llama3 8b
llama3_8b_path = '/model/Aaronzhu/LLAMA3'
llama3_8b_path_tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llama3_8b_path,
    trust_remote_code=True,
    padding_side='right')
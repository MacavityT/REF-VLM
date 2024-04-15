gptgen_train_common_cfg = dict(
    type='GPT4Gen',
    text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/GPT4GEN_BoxCoT_train.jsonl',
    image_folder=r'/data/Aaronzhu/DatasetStage1/flickr30k/flickr30k-images',
)

train_gptgen_variant = dict(
    gpt4gen_qa=dict(**gptgen_train_common_cfg, version='a', template_name=r"VQA"),
    gpt4gen_qc=dict(**gptgen_train_common_cfg, version='c', template_name=r"VQA_CoT"),
    gpt4gen_qbc=dict(**gptgen_train_common_cfg, version='bc', template_name=r"VQA_BCoT"),

    gpt4gen_rd_qbc=dict(
        type=gptgen_train_common_cfg['type'],
        image_folder=gptgen_train_common_cfg['image_folder'],
        text_path='/data/Aaronzhu/DatasetStage1/Shikra/GPT4GEN_RD_BoxCoT_train.jsonl',
        version='bc',
        template_name=r"VQA_BCoT"),
)

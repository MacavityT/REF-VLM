gptgen_train_common_cfg = dict(
    type='GPT4Gen',
    text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/GPT4GEN_BoxCoT_train.jsonl',
    image_folder=r'/data/Aaronzhu/DatasetStage1/flickr30k/flickr30k-images',
    stage=2,
)

train_gptgen_variant = dict(
    gpt4gen_qa=dict(
        **gptgen_train_common_cfg, 
        version='a', 
        map_placeholders=dict(
            input=["<boxes>"],
        ),
        template_name=r"VQA",
        offline_processed_text_folder = '',
    ),
    gpt4gen_qc=dict(
        **gptgen_train_common_cfg, 
        version='c', 
        map_placeholders=dict(
            input=["<boxes>"],
        ),
        template_name=r"VQA_CoT",
        offline_processed_text_folder = '',
    ),
    gpt4gen_qbc=dict(
        **gptgen_train_common_cfg, 
        version='bc', 
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<boxes>"],
        ),
        template_name=r"VQA_BCoT",
        offline_processed_text_folder = '',
    ),
    gpt4gen_rd_qbc=dict(
        type=gptgen_train_common_cfg['type'],
        image_folder=gptgen_train_common_cfg['image_folder'],
        text_path='/data/Aaronzhu/DatasetStage1/Shikra/GPT4GEN_RD_BoxCoT_train_modified.jsonl',
        offline_processed_text_folder = '',
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<boxes>"],
        ),
        stage=2,
        version='bc',
        template_name=r"VQA_BCoT",
    ),
)

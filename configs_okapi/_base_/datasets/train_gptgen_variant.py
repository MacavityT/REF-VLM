gptgen_train_common_cfg = dict(
    type='GPT4Gen',
    filename=r'{{fileDirname}}/../../../data/GPT4GEN_BoxCoT_train.jsonl',
    image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
)

train_gptgen_variant = dict(
    gpt4gen_qa=dict(**gptgen_train_common_cfg, version='a', template_file=r"VQA"),
    gpt4gen_qc=dict(**gptgen_train_common_cfg, version='c', template_file=r"VQA_CoT"),
    gpt4gen_qbc=dict(**gptgen_train_common_cfg, version='bc', template_file=r"VQA_BCoT"),

    gpt4gen_rd_qbc=dict(
        type=gptgen_train_common_cfg['type'],
        image_folder=gptgen_train_common_cfg['image_folder'],
        filename='{{fileDirname}}/../../../data/GPT4GEN_RD_BoxCoT_train.jsonl',
        version='bc',
        template_file=r"VQA_BCoT"),
)

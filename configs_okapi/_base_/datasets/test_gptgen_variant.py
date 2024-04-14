GPTGEN_TEST_COMMON_CFG = dict(
    type='GPT4Gen',
    filename=r'/data/Aaronzhu/DatasetStage1/Shikra/GPT4GEN_BoxCoT_test.jsonl',
    image_folder=r'/data/Aaronzhu/DatasetStage1/flickr30k/flickr30k-images',
)

test_gptgen_variant = dict(
    GPT4GEN_QA=dict(**GPTGEN_TEST_COMMON_CFG, version='a', template_file=r"VQA"),
    GPT4GEN_QC=dict(**GPTGEN_TEST_COMMON_CFG, version='c', template_file=r"VQA_CoT"),
    GPT4GEN_QBC=dict(**GPTGEN_TEST_COMMON_CFG, version='bc', template_file=r"VQA_BCoT"),
)

vqav2_test_common_cfg = dict(
    type='VQAv2Dataset',
    image_folder=r'/data/Aaronzhu/DatasetStage1/VQAv2/real_images/train2014',
    template_name=r"VQA",
)

test_vqav2_variant = dict(
    vqav2_val=dict(
        **vqav2_test_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/v2_OpenEnded_mscoco_val2014_questions.jsonl',
    ),
    vqav2_testdev=dict(
        **vqav2_test_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/v2_OpenEnded_mscoco_test-dev2015_questions.jsonl',
        has_annotation=False,
    ),
    vqav2_test=dict(
        **vqav2_test_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/v2_OpenEnded_mscoco_test2015_questions.jsonl',
        has_annotation=False,
    ),
)

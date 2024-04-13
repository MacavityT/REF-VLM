vqav2_test_common_cfg = dict(
    type='VQAv2Dataset',
    image_folder=r'zz1424:s3://publicdataset_49/VQAv2/unzip/',
    template_file=r"VQA",
)

test_vqav2_variant = dict(
    vqav2_val=dict(
        **vqav2_test_common_cfg,
        filename=r'{{fileDirname}}/../../../data/v2_OpenEnded_mscoco_val2014_questions.jsonl',
    ),
    vqav2_testdev=dict(
        **vqav2_test_common_cfg,
        filename=r'{{fileDirname}}/../../../data/v2_OpenEnded_mscoco_test-dev2015_questions.jsonl',
        has_annotation=False,
    ),
    vqav2_test=dict(
        **vqav2_test_common_cfg,
        filename=r'{{fileDirname}}/../../../data/v2_OpenEnded_mscoco_test2015_questions.jsonl',
        has_annotation=False,
    ),
)

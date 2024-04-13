vqav2_train_common_cfg = dict(
    type='VQAv2Dataset',
    filename=r'{{fileDirname}}/../../../data/v2_OpenEnded_mscoco_train2014_questions.jsonl',
    image_folder=r'zz1424:s3://publicdataset_49/VQAv2/unzip/',
    template_file=r"VQA",
)

train_vqav2_variant = dict(
    vqav2_train=dict(**vqav2_train_common_cfg),
)
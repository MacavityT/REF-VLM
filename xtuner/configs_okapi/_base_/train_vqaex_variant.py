vqaex_train_common_cfg = dict(
    type='VQAEXDataset',
    image_folder=r'zz1424:s3://visual_grounding/academic_data/refer/images/mscoco/images/',
    template_file=r"VQA_CoT",
)

train_vqaex_variant = dict(
    vqae_train=dict(
        **vqaex_train_common_cfg,
        is_e_dataset=True,
        filename=r'{{fileDirname}}/../../../data/vqa_E_train.jsonl',
    ),
    vqax_train=dict(
        **vqaex_train_common_cfg,
        is_e_dataset=False,
        filename=r'{{fileDirname}}/../../../data/vqa_X_train.jsonl',
    ),
)

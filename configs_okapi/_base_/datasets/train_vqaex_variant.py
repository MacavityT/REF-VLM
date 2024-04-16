vqaex_train_common_cfg = dict(
    type='VQAEXDataset',
    image_folder=r'/data/Aaronzhu/DatasetStage1/VQAv2/real_images/train2014',
    image_info_folder=r'/data/Aaronzhu/DatasetStage1/Shikra/shape/vqav2_train_shape.jsonl',
    template_name=r"VQA_CoT",
)

train_vqaex_variant = dict(
    vqae_train=dict(
        **vqaex_train_common_cfg,
        is_e_dataset=True,
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/vqa_E_train.jsonl',
    ),
    vqax_train=dict(
        **vqaex_train_common_cfg,
        is_e_dataset=False,
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/vqa_X_train.jsonl',
    ),
)

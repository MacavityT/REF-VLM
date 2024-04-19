vqav2_train_common_cfg = dict(
    type='VQAv2Dataset',
    text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/v2_OpenEnded_mscoco_train2014_questions.jsonl',
    image_info_folder='/data/Aaronzhu/DatasetStage1/Shikra/shape/vqav2_train_shape.jsonl',
    image_folder=r'/data/Aaronzhu/DatasetStage1/VQAv2/real_images/train2014',
    offline_processed_image_folder = '',
    template_name=r"VQA",
)

train_vqav2_variant = dict(
    vqav2_train=dict(
        **vqav2_train_common_cfg,
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage1/offline_text_datasets/vqav2_train'
    ),
)
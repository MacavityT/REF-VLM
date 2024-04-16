vqav2_train_common_cfg = dict(
    type='VQAv2Dataset',
    text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/v2_OpenEnded_mscoco_train2014_questions.jsonl',
    image_info_folder='/data/Aaronzhu/DatasetStage1/Shikra/shape/vqav2_train_shape.jsonl',
    image_folder=r'/data/Aaronzhu/DatasetStage1/VQAv2/real_images/train2014',
    template_name=r"VQA",
)

train_vqav2_variant = dict(
    vqav2_train=dict(**vqav2_train_common_cfg),
)
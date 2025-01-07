chinese_text_common_cfg = dict(
    type='OCRCNDataset',
    image_folder=r'',
    template_name=r"OCR_CN",
)

train_chinese_text_variant = dict(
    text_recognition_train=dict(
        **chinese_text_common_cfg,
        text_path=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/chinese_text_recognition/data/train-00000-of-00001-d4f8f143c5f9fcdd.parquet',
    ),
    text_train=dict(
        **chinese_text_common_cfg,
        text_path=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/ChineseTest/train-00056-of-00087-6099752c7388fc48.parquet',
    ),
)
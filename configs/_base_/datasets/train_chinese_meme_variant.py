chinese_meme_common_cfg = dict(
    type='MEMEDataset',
    template_name=r"image_cap_cn",
)

train_chinese_meme_variant = dict(
    cn_bqb_yi=dict(
        **chinese_meme_common_cfg,
        image_folder=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/chinese-meme-description-dataset/ChineseBQB/images',
        text_path=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/chinese-meme-description-dataset/ChineseBQB_picked_yivision.jsonl',
    ),
    cn_bqb_claude=dict(
        **chinese_meme_common_cfg,
        image_folder=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/chinese-meme-description-dataset/ChineseBQB/images',
        text_path=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/chinese-meme-description-dataset/ChineseBQB-Claude-3-5-sonnet-20240620.jsonl',
    ),
    cn_bqb_gemini=dict(
        **chinese_meme_common_cfg,
        image_folder=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/chinese-meme-description-dataset/ChineseBQB/images',
        text_path=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/chinese-meme-description-dataset/ChineseBQB-picked-gemini-1.5-pro-latest_cleaned.jsonl',
    ),
    cn_emo_claude=dict(
        **chinese_meme_common_cfg,
        image_folder=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/chinese-meme-description-dataset/emo',
        text_path=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/chinese-meme-description-dataset/emo-visual-data-Claude-3-5-sonnet-20240620.jsonl',
    ),
    cn_emo_gemini=dict(
        **chinese_meme_common_cfg,
        image_folder=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/chinese-meme-description-dataset/emo',
        text_path=r'/data/Aaronzhu/ChineseDatasets/llava-instruct-chinese/chinese-meme-description-dataset/emo-visual-data-gemini-1_5_pro.jsonl',
    ),
)
cvlue_common_cfg = dict(
    image_folder=r'/data/Aaronzhu/ChineseDatasets/CVLUE/',
)

train_cvlue_variant = dict(
    cvlue_dialog=dict(
        **cvlue_common_cfg,
        type='CVLUEDialogueDataset',
        text_path=r'/data/Aaronzhu/ChineseDatasets/CVLUE/CVLUE_train_annotation/train_VD.json',
    ),
    cvlue_vqa=dict(
        **cvlue_common_cfg,
        type='CVLUEVQADataset',
        text_path=r'/data/Aaronzhu/ChineseDatasets/CVLUE/CVLUE_train_annotation/train_VQA.json',
        template_name=r'VQA_CN',
    ),
    cvlue_caption=dict(
        **cvlue_common_cfg,
        type='CVLUECaptionDataset',
        text_path=r'/data/Aaronzhu/ChineseDatasets/CVLUE/CVLUE_train_annotation/train_caption.jsonl',
        template_name=r'image_cap_cn',
    ),
    cvlue_rec=dict(
        **cvlue_common_cfg,
        type='CVLUERECDataset',
        text_path=r'/data/Aaronzhu/ChineseDatasets/CVLUE/CVLUE_train_annotation/train_VG.json',
        template_name=r'REC_CN',
    ),


)
voc2010_train_common_cfg = dict(
    text_path=r'/data/Aaronzhu/DatasetStage2and3/VOCdevkit/VOC2010/trainval_merged.json',  
    image_folder=r'/data/Aaronzhu/DatasetStage2and3/pascal_ctx_d2/images/training',
    offline_processed_text_folder = '',
    offline_processed_image_folder = '',
    )
voc2012_train_common_cfg = dict(
    type='PscalVocDataset',
    text_path=r'/data/Aaronzhu/DatasetStage2and3/VOCdevkit/VOC2010/trainval_merged.json',  
    image_folder=r'/data/Aaronzhu/DatasetStage2and3/pascal_ctx_d2/images/training',
    offline_processed_text_folder = '',
    offline_processed_image_folder = '',
    )
train_voc_variant = dict(
    train_voc59_with_semantic=dict(
        **voc2010_train_common_cfg,
        type='PscalVoc59Dataset',
        gt_info=r'/data/Aaronzhu/DatasetStage2and3/pascal_ctx_d2/annotations_ctx59/training/',
        template_name=r'Cond_SEG',
    ),
    train_voc459_with_semantic=dict(
        **voc2010_train_common_cfg,
        type='PscalVoc459Dataset',
        gt_info=r'/data/Aaronzhu/DatasetStage2and3/pascal_ctx_d2/annotations_ctx459/training/',
        template_name=r'Cond_SEG',
    ),
    train_voc20_with_semantic=dict(
        **voc2012_train_common_cfg,
        gt_info=r'/data/Aaronzhu/DatasetStage2and3/pascal_voc_d2/annotations_pascal20/training/',
        template_name=r'Cond_SEG',
    ),
)
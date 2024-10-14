ade20k_train_common_cfg = dict(
    type='ADE20k',
    text_path=r'/data/Aaronzhu/DatasetStage2and3/ADE20k/ADEChallengeData2016/ade20k_instance_train.json',  
    image_folder=r'/data/Aaronzhu/DatasetStage2and3/ADE20k/ADEChallengeData2016/images/training',
    )

train_ade20k_variant = dict(
    train_ade20_with_semantic=dict(
        **ade20k_train_common_cfg,
        gt_info=r'/data/Aaronzhu/DatasetStage2and3/ADE20k/ADEChallengeData2016/annotations_semantic/training',
        target_type = 'semantic',
        template_name=r'Cond_SEG',
        map_placeholders=dict(
            output=["<masks>"],
        ),     
    ),
    train_ade20_with_instance=dict(
        **ade20k_train_common_cfg,
        gt_info=r'/data/Aaronzhu/DatasetStage2and3/ADE20k/ADEChallengeData2016/annotations_instance/training',
        target_type = 'instance',
        template_name=r'SEG',
        map_placeholders=dict(
            output=["<masks>"],
        ),
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/ADE20k/ADEChallengeData2016/offline',
    ),
    train_ade20_with_panotic=dict(
        **ade20k_train_common_cfg,
        gt_info=r'/data/Aaronzhu/DatasetStage2and3/ADE20k/ADEChallengeData2016/ade20k_panoptic_train',
        target_type = 'panotic',
        template_name=r'Cond_SEG',
        map_placeholders=dict(
            output=["<masks>"],
        ),  
    ),
)
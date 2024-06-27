cityscapes_train_common_cfg = dict(
    text_path=r'/data/Aaronzhu/DatasetStage2and3/cityscapes/gtFine/train',  
    image_folder=r'/data/Aaronzhu/DatasetStage2and3/cityscapes/leftImg8bit/train',
    placeholders=('<image>',),
    ratio=0.3,
    )


train_cityscapes_variant = dict(
    train_cityscapes_semantic=dict(
        **cityscapes_train_common_cfg,
        split='train',
        type='CityscapesSemantic',
        target_type = 'semantic',
        template_name=r'Cond_SEG',
        map_placeholders=dict(
            output=["<masks>"],
        ),     
    ),
    train_cityscapes_instance=dict(
        **cityscapes_train_common_cfg,
        type='CityscapesInstance',
        split='train',
        target_type='instance',
        template_name=r'SEG',
        map_placeholders=dict(
            output=["<masks>"],
        ),
        offline_processed_text_folder='',     
    ),
)
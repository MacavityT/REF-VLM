interact_train_common_cfg=dict(
    text_path=r'/data/Aaronzhu/DatasetStage2and3/COCO_interactive/coco_interactive_train_psalm.json',
    image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/train2017',
)

interact_train_single_common_cfg=dict(
    text_path=r'/data/Aaronzhu/DatasetStage2and3/COCO_interactive/interactive_train_single.json',
    image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/train2017',
)

train_interact_variant = dict(
    interact_mask=dict(
        **interact_train_common_cfg,
        type='COCOInteract', 
        template_name=r'Region_SEG',
        version='s',
        max_conv_length=6,
        placeholders=('<image>','<region>'),
        map_placeholders=dict(
            input=["<masks>"],
            output=["<masks>"],
        ),
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/COCO_interactive/mask_offline_new'
    ),
    interact_box=dict(
        **interact_train_common_cfg,
        type='COCOInteract', 
        template_name=r'Region_DET',
        version='d',
        max_conv_length=6,
        placeholders=('<image>','<region>'),
        map_placeholders=dict(
            input=["<masks>"],
            output=["<boxes>"],
        ),
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/COCO_interactive/offline_box' 
    ),
    interact_reg=dict(
        **interact_train_common_cfg,
        type='COCOInteract', 
        template_name=r'REG_SEG',
        version='r',
        max_conv_length=6,
        placeholders=('<image>','<masks>'),
        map_placeholders=dict(
            input=["<masks>"],
        ),
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/COCO_interactive/offline_reg'      
    ),
    interact_reg_single=dict(
        **interact_train_common_cfg,
        type='COCOInteractSingle', 
        template_name=r'REG_SEG',
        split='train',
        version='r',
        placeholders=('<image>','<masks>'),
        map_placeholders=dict(
            input=["<masks>"],
        ),       
    ),
    interact_box_single=dict(
        **interact_train_common_cfg,
        type='COCOInteractSingle', 
        template_name=r'Region_DET',
        split='train',
        version='d',
        placeholders=('<image>','<region>'),
        map_placeholders=dict(
            input=["<masks>"],
            output=["<boxes>"],
        ),    
    ),
    interact_mask_single=dict(
        **interact_train_common_cfg,
        type='COCOInteractSingle', 
        template_name=r'Region_SEG',
        split='train',
        version='s',
        placeholders=('<image>','<region>'),
        map_placeholders=dict(
            input=["<masks>"],
            output=["<masks>"],
        ),    
    ),
    interact_box_finetune=dict(
        **interact_train_single_common_cfg,
        type='COCOInteractSingleTask', 
        template_name=r'Region_SEG',
        split='train',
        version='box',
        placeholders=('<image>','<region>'),
        map_placeholders=dict(
            input=["<masks>"],
            output=["<masks>"],
        ),    
    ),
    interact_mask_finetune=dict(
        **interact_train_single_common_cfg,
        type='COCOInteractSingleTask', 
        template_name=r'Region_SEG',
        split='train',
        version='mask',
        placeholders=('<image>','<region>'),
        map_placeholders=dict(
            input=["<masks>"],
            output=["<masks>"],
        ),    
    ),
    interact_point_finetune=dict(
        **interact_train_single_common_cfg,
        type='COCOInteractSingleTask', 
        template_name=r'Region_SEG',
        split='train',
        version='point',
        placeholders=('<image>','<region>'),
        map_placeholders=dict(
            input=["<masks>"],
            output=["<masks>"],
        ),    
    ),
    interact_scribble_finetune=dict(
        **interact_train_single_common_cfg,
        type='COCOInteractSingleTask', 
        template_name=r'Region_SEG',
        split='train',
        version='scribble',
        placeholders=('<image>','<region>'),
        map_placeholders=dict(
            input=["<masks>"],
            output=["<masks>"],
        ),    
    ),
)
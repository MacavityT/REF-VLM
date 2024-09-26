interact_test_common_cfg=dict(
    type='COCOInteractSingle',
    text_path=r'/data/Aaronzhu/DatasetStage2and3/COCO_interactive/coco_interactive_val_psalm.json',
    image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/val2017',
)

interact_train_single_common_cfg=dict(
    text_path=r'/data/Aaronzhu/DatasetStage2and3/COCO_interactive/interactive_val_single.json',
    image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/val2017',
)

test_interact_variant = dict(
    interact_mask=dict(
        **interact_test_common_cfg, 
        template_name=r'Region_SEG',
        version='s',
        placeholders=('<image>','<region>'),
        map_placeholders=dict(
            input=["<masks>"],
            output=["<masks>"],
        ),
        offline_processed_text_folder=''     
    ),
    interact_box=dict(
        **interact_test_common_cfg, 
        template_name=r'Region_DET',
        version='d',
        placeholders=('<image>','<region>'),
        map_placeholders=dict(
            input=["<masks>"],
            output=["<boxes>"],
        ),
        offline_processed_text_folder=''           
    ),
    interact_reg=dict(
        **interact_test_common_cfg, 
        template_name=r'REG_SEG',
        version='r',
        placeholders=('<image>','<masks>'),
        map_placeholders=dict(
            input=["<masks>"],
        ),
        offline_processed_text_folder=''           
    ),
    interact_box_finetune=dict(
        **interact_train_single_common_cfg,
        type='COCOInteractSingleTask', 
        template_name=r'Region_SEG',
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
        version='scribble',
        placeholders=('<image>','<region>'),
        map_placeholders=dict(
            input=["<masks>"],
            output=["<masks>"],
        ),    
    )
)
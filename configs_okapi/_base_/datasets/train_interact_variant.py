from xtuner.utils.constants import BOXES_PLACEHOLDER,MASKS_PLACEHOLDER,REGION_PLACEHOLDER

interact_train_common_cfg=dict(
    type='COCOInteract',
    text_path=r'/data/Aaronzhu/DatasetStage2and3/COCO_interactive/coco_interactive_train_psalm.json',
    image_folder=r'/data/Aaronzhu/DatasetStage2and3/COCO_interactive/train2017',
)


train_interact_variant = dict(
    interact_mask=dict(
        **interact_train_common_cfg, 
        template_name=r'Region_SEG',
        version='s',
        max_conv_length=6,
        map_placeholders=dict(
            input=[REGION_PLACEHOLDER],
            output=[MASKS_PLACEHOLDER],
        ),       
    ),
    interact_box=dict(
        **interact_train_common_cfg, 
        template_name=r'Region_DET',
        version='d',
        max_conv_length=6,
        map_placeholders=dict(
            input=[REGION_PLACEHOLDER],
            output=[BOXES_PLACEHOLDER],
        ),       
    )
)
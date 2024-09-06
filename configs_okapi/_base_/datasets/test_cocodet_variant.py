cocodet_test_common_cfg = dict(
    type='COCOREMDataset',
    task_type='box',
    template_name=r'DET',
    placeholders=('<image>',),
    map_placeholders=dict(
        output=["<boxes>"],
    ),
)


test_cocodet_variant=dict(
    coco_rem_box_val=dict(
        **cocodet_test_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/COCO-ReM/instances_valrem.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/COCO-ReM/train2017',
        offline_processed_text_folder='',
    ),
    coco_2014_box_val=dict(
        **cocodet_test_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/annotations/instances_val2014.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/annotations/instances_val2014.json',
        offline_processed_text_folder='',
    ),
    coco_2017_box_val=dict(
        **cocodet_test_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/annotations/instances_val2017.json',
        image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/val2017',
        offline_processed_text_folder='',
    ),
)

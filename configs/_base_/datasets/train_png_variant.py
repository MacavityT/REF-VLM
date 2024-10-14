png_train_common_cfg=dict(
    type='PNGDataset', 
    text_path=r'/data/Aaronzhu/DatasetStage2and3/PNG/png_coco_train2017.json',
    # anno_path=r'/data/Aaronzhu/DatasetStage2and3/PNG/png_masks.pkl',
    anno_path=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/annotations/panoptic_train2017.json',
    anno_img_dir=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/panoptic_train2017',
    image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2017/train2017',
)


train_png_variant = dict(
    png_reg=dict(
        **png_train_common_cfg,
        template_name=r"REG_SEG",
        version='reg',
        placeholders=('<image>',),
        map_placeholders=dict(
            input=["<masks>"],
        ),
        offline_processed_text_folder=''     
    ),
    png_gcg=dict(
        **png_train_common_cfg,
        template_name=r"flickr30k_SEG",
        version='gcg',
        placeholders=('<image>',),
        map_placeholders=dict(
            output=["<masks>"],
        ),
        offline_processed_text_folder='/data/Aaronzhu/DatasetStage2and3/PNG/offline_gcg'           
    ),

)


cocogcg_common_cfg=dict(
    type='COCOGCG',
    image_folder='/data/Aaronzhu/GranD/GLaMM_data/GranD-f/GranDf_HA_images/val_test',
    template_name=r"flickr30k_SEG",
    placeholders=('<image>',),
    map_placeholders=dict(
        output=["<masks>"],
    ), 
)


test_cocogcg_variant = dict(
    cocogcg_val=dict(
        **cocogcg_common_cfg,
        text_path='/data/Aaronzhu/GranD/GLaMM_data/GranD-f/annotations/val_test/val_test_val_gcg_coco_caption_gt.json',
        mask_path='/data/Aaronzhu/GranD/GLaMM_data/GranD-f/annotations/val_test/val_gcg_coco_mask_gt.json'
    ),
    cocogcg_test=dict(
        **cocogcg_common_cfg,
        text_path='/data/Aaronzhu/GranD/GLaMM_data/GranD-f/annotations/val_test/val_test_test_gcg_coco_caption_gt.json',
        mask_path='/data/Aaronzhu/GranD/GLaMM_data/GranD-f/annotations/val_test/test_gcg_coco_mask_gt.json',
    ),
)
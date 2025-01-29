grand_test_common_cfg = dict(
    type='GranDDataset',
    text_path=r'/data/Aaronzhu/GranD/GranD/Jsons',
    image_folder=r'/data/Aaronzhu/SA-1B/OpenDataLab___SA-1B/images',
    stage=2,
)

test_grand_variant = dict(
    grand_cond_d=dict(
        **grand_test_common_cfg, 
        version='cond_d', 
        max_conv_length=6,
        template_name=r"Cond_DET",
        map_placeholders=dict(
            output=["<boxes>"],
        ),     
        placeholders=('<image>','<cls>'),
        offline_processed_text_folder='/data/Aaronzhu/GranD/GranD/offline_single_task/offline_rec_multi_objects_single',     
    ),
    grand_cond_s=dict(
        **grand_test_common_cfg,
        version='cond_s', 
        max_conv_length=6,
        template_name=r"Cond_SEG",
        map_placeholders=dict(
            output=["<masks>"],
        ), 
        placeholders=('<image>','<cls>'),
        offline_processed_text_folder='/data/Aaronzhu/GranD/GranD/offline_single_task/offline_res_multi_objects_single',            
    ),
)
nocaps_test_common_cfg = dict(
    type='NoCapsDataset',
    template_name=r'image_cap'
)

test_nocaps_variant = dict(
    nocaps_val=dict(
        **nocaps_test_common_cfg,
        image_folder=r'/data/Aaronzhu/DatasetStage2and3/Nocaps/images',
        text_path='/data/Aaronzhu/DatasetStage2and3/Nocaps/nocaps_val_4500_captions.json',
    ),
)
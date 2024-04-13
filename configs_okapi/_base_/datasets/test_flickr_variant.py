flickr_test_common_cfg = dict(
    type='FlickrDataset',
    image_folder=r'zz1424:s3://production-public-flickr_image/Flickr_Image/unzip/flickr30k_images/flickr30k_images',
    max_dynamic_size=None,
)

test_flickr_variant = dict(
    flickr_eval_with_box=dict(
        **flickr_test_common_cfg,
        filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_eval.jsonl',
        template_file=r'flickr30k',
    ),
    flickr_eval_without_box=dict(
        **flickr_test_common_cfg,
        filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_eval.jsonl',
        template_file=r'image_cap',
    ),
    flickr_test_with_box=dict(
        **flickr_test_common_cfg,
        filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_test.jsonl',
        template_file=r'flickr30k',
    ),
    flickr_test_without_box=dict(
        **flickr_test_common_cfg,
        filename=r'{{fileDirname}}/../../../data/CWB_flickr30k_test.jsonl',
        template_file=r'image_cap',
    ),
)

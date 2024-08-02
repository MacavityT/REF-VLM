flickr_test_common_cfg = dict(
    image_folder=r'/data/Aaronzhu/DatasetStage1/flickr30k/flickr30k-images',
    max_dynamic_size=None,
)

test_flickr_variant = dict(
    flickr_eval_with_box=dict(
        **flickr_test_common_cfg,
        type='FlickrDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CWB_flickr30k_eval.jsonl',
        template_name=r'flickr30k',
    ),
    flickr_eval_without_box=dict(
        **flickr_test_common_cfg,
        type='FlickrCaptionDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CWB_flickr30k_eval.jsonl',
        template_name=r'image_cap',
    ),
    flickr_test_with_box=dict(
        **flickr_test_common_cfg,
        type='FlickrDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CWB_flickr30k_test.jsonl',
        template_name=r'flickr30k',
    ),
    flickr_test_without_box=dict(
        **flickr_test_common_cfg,
        type='FlickrCaptionDataset',
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CWB_flickr30k_test.jsonl',
        template_name=r'image_cap',
    ),
)

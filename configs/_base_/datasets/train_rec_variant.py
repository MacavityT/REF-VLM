res_train_common_cfg=dict(
    type='RECDataset',
    image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
    template_name=r'REC',
    target=False,
    map_placeholders=dict(
        output=["<boxes>"],
    )
)

train_rec_variant = dict(
    rec_refcoco=dict(
        **res_train_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/Refcoco/refcoco/train.jsonl',
    ),
    rec_refcocoa=dict(
        **res_train_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/Refcoco/refcoco+/train.jsonl',
    ),
    rec_refcocog=dict(
        **res_train_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/Refcoco/refcocog/train.jsonl',
    ),
)

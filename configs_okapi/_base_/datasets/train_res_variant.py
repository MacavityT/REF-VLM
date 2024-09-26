res_train_common_cfg = dict(
    type='RESDataset',
    image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
    template_name=r'RES',
    # target=True,
    max_dynamic_size=None,
    map_placeholders=dict(
        output=["<masks>"],
    ), 
)

train_res_variant = dict(
    res_refcoco=dict(
        **res_train_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/Refcoco/refcoco/train.jsonl',
    ),
    res_refcocoa=dict(
        **res_train_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/Refcoco/refcoco+/train.jsonl',
    ),
    res_refcocog=dict(
        **res_train_common_cfg,
        text_path=r'/data/Aaronzhu/DatasetStage1/Refcoco/refcocog/train.jsonl',
    ),
)

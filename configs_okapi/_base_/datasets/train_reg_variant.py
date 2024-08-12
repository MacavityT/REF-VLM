reg_train_common_cfg = dict(
    type='REGDataset',
    image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
    max_dynamic_size=None,
)

train_reg_variant = dict(
    reg_refcocog_train_box=dict(
        **reg_train_common_cfg,
        template_name=r'REG',
        version='box',
        text_path=r'/data/Aaronzhu/DatasetStage1/Refcoco/refcocog/train.jsonl',
        map_placeholders=dict(
            input=["<boxes>"],
        ),     
        placeholders=('<image>','<objs>'),
    ),
    reg_refcocog_train_mask=dict(
        **reg_train_common_cfg,
        template_name=r'REG_SEG',
        version='mask',
        text_path=r'/data/Aaronzhu/DatasetStage1/Refcoco/refcocog/train.jsonl',
        map_placeholders=dict(
            input=["<masks>"],
        ), 
        placeholders=('<image>',"<masks>"),
    ),
)

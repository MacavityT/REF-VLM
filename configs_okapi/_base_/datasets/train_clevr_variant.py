clevr_train_common_cfg = dict(
    type='ClevrDataset',
    text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/CLEVR_train_questions_with_ans.jsonl',
    image_folder=r'zz1424:s3://publicdataset_33/CLEVR/unzip/CLEVR_v1.0/images/train',
    scene_graph_file=r"/data/Aaronzhu/DatasetStage2and3/CLEVR/CLEVR_v1.0/images/train",
)

train_clevr_variant = dict(
    clevr_a=dict(
        **clevr_train_common_cfg,
        version='q-a',
        template_name=r"VQA.json",
    ),
    clevr_s=dict(
        **clevr_train_common_cfg,
        version='q-s',
        template_name=r"VQA_CoT.json",
    ),
    clevr_bs=dict(
        **clevr_train_common_cfg,
        version='q-bs',
        template_name=r"VQA_PCoT.json",
    ),
)

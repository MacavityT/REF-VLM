clevr_test_common_cfg = dict(
    type='ClevrDataset',
    filename=r'/data/Aaronzhu/DatasetStage1/Shikra/CLEVR_val_questions_with_ans.jsonl',
    image_folder=r'/data/Aaronzhu/DatasetStage2and3/CLEVR/CLEVR_v1.0/images/val',
    scene_graph_file=None,
)

test_clevr_variant = dict(
    clevr_a_val=dict(
        **clevr_test_common_cfg,
        version='q-a',
        template_file=r"VQA",
    ),
    clevr_s_val=dict(
        **clevr_test_common_cfg,
        version='q-a',
        template_file=r"VQA_CoT",
    ),
    clevr_bs_val=dict(
        **clevr_test_common_cfg,
        version='q-a',
        template_file=r"VQA_PCoT",
    ),
    
)

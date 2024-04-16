gqa_test_common_cfg = dict(
    type='GQADataset',
    image_folder=r'/data/Aaronzhu/DatasetStage2and3/llava-instruct/images/gqa/images',
    scene_graph_file=None,
    scene_graph_index=None,
)

# use standard q-a mode
test_gqa_variant = dict(
    gqa_q_a_balanced=dict(
        **gqa_test_common_cfg, version="q-a", template_name=r"VQA",
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/gqa_testdev_balanced_questions.jsonl'
    ),
    gqa_q_c_balanced=dict(
        **gqa_test_common_cfg, version="q-a", template_name=r"VQA_CoT",
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/gqa_testdev_balanced_questions.jsonl'
    ),
    gqa_q_bc_balanced=dict(
        **gqa_test_common_cfg, version="q-a", template_name=r"VQA_BCoT",
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/gqa_testdev_balanced_questions.jsonl'
    ),

    gqa_q_a=dict(
        **gqa_test_common_cfg, version="q-a", template_name=r"VQA",
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/gqa_testdev_all_questions.jsonl',
    ),
    gqa_q_c=dict(
        **gqa_test_common_cfg, version="q-a", template_name=r"VQA_CoT",
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/gqa_testdev_all_questions.jsonl',
    ),
    gqa_q_bc=dict(
        **gqa_test_common_cfg, version="q-a", template_name=r"VQA_BCoT",
        text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/gqa_testdev_all_questions.jsonl',
    ),
)

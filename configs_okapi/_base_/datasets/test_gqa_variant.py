gqa_test_common_cfg = dict(
    type='GQADataset',
    image_folder=r'zz1424:s3://publicdataset_11/GQA/unzip/images',
    scene_graph_file=None,
    scene_graph_index=None,
)

# use standard q-a mode
test_gqa_variant = dict(
    gqa_q_a_balanced=dict(
        **gqa_test_common_cfg, version="q-a", template_file=r"VQA",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_balanced_questions.jsonl'
    ),
    gqa_q_c_balanced=dict(
        **gqa_test_common_cfg, version="q-a", template_file=r"VQA_CoT",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_balanced_questions.jsonl'
    ),
    gqa_q_bc_balanced=dict(
        **gqa_test_common_cfg, version="q-a", template_file=r"VQA_BCoT",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_balanced_questions.jsonl'
    ),

    gqa_q_a=dict(
        **gqa_test_common_cfg, version="q-a", template_file=r"VQA",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_all_questions.jsonl',
    ),
    gqa_q_c=dict(
        **gqa_test_common_cfg, version="q-a", template_file=r"VQA_CoT",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_all_questions.jsonl',
    ),
    gqa_q_bc=dict(
        **gqa_test_common_cfg, version="q-a", template_file=r"VQA_BCoT",
        filename=r'{{fileDirname}}/../../../data/gqa_testdev_all_questions.jsonl',
    ),
)

gqa_train_common_cfg = dict(
    type='GQADataset',
    text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/gqa_question_balanced_with_cot.jsonl',
    image_folder=r'/data/Aaronzhu/DatasetStage2and3/llava-instruct/images/gqa/images',
    scene_graph_file=r"/data/Aaronzhu/DatasetStage1/Shikra/gqa_scene_graph_data.jsonl",
    scene_graph_index=r"/data/Aaronzhu/DatasetStage1/Shikra/gqa_scene_graph_index.json",
)

train_gqa_variant = dict(
    gqa_q_a=dict(**gqa_train_common_cfg, version="q-a", template_name=r"VQA"),
    gqa_q_c=dict(**gqa_train_common_cfg, version="q-c", template_name=r"VQA_CoT"),
    gqa_q_bc=dict(**gqa_train_common_cfg, version="q-bc", template_name=r"VQA_BCoT"),
    gqa_q_s=dict(**gqa_train_common_cfg, version="q-s", template_name=r"VQA_CoT"),
    gqa_q_bs=dict(**gqa_train_common_cfg, version="q-bs", template_name=r"VQA_BCoT"),
    gqa_q_l=dict(**gqa_train_common_cfg, version="q-l", template_name=r"VQA_CoT"),
    gqa_q_bl=dict(**gqa_train_common_cfg, version="q-bl", template_name=r"VQA_BCoT"),

    gqa_qb_a=dict(**gqa_train_common_cfg, version="qb-a", template_name=r"VQA"),
    gqa_qb_c=dict(**gqa_train_common_cfg, version="qb-c", template_name=r"VQA_CoT"),
    gqa_qb_bc=dict(**gqa_train_common_cfg, version="qb-bc", template_name=r"VQA_BCoT"),
    gqa_qb_s=dict(**gqa_train_common_cfg, version="qb-s", template_name=r"VQA_CoT"),
    gqa_qb_bs=dict(**gqa_train_common_cfg, version="qb-bs", template_name=r"VQA_BCoT"),
    gqa_qb_l=dict(**gqa_train_common_cfg, version="qb-l", template_name=r"VQA_CoT"),
    gqa_qb_bl=dict(**gqa_train_common_cfg, version="qb-bl", template_name=r"VQA_BCoT"),

    gqa_qbp_a=dict(**gqa_train_common_cfg, version="qbp-a", template_name=r"VQA"),
    gqa_qbp_c=dict(**gqa_train_common_cfg, version="qbp-c", template_name=r"VQA_CoT"),
    gqa_qbp_bc=dict(**gqa_train_common_cfg, version="qbp-bc", template_name=r"VQA_BCoT"),
    gqa_qbp_s=dict(**gqa_train_common_cfg, version="qbp-s", template_name=r"VQA_CoT"),
    gqa_qbp_bs=dict(**gqa_train_common_cfg, version="qbp-bs", template_name=r"VQA_BCoT"),
    gqa_qbp_l=dict(**gqa_train_common_cfg, version="qbp-l", template_name=r"VQA_CoT"),
    gqa_qbp_bl=dict(**gqa_train_common_cfg, version="qbp-bl", template_name=r"VQA_BCoT"),
)

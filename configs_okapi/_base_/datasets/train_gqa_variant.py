gqa_train_common_cfg = dict(
    type='GQADataset',
    filename=r'{{fileDirname}}/../../../data/gqa_question_balanced_with_CoT.jsonl',
    image_folder=r'zz1424:s3://publicdataset_11/GQA/unzip/images',
    scene_graph_file=r"{{fileDirname}}/../../../data/gqa_scene_graph_data.jsonl",
    scene_graph_index=r"{{fileDirname}}/../../../data/gqa_scene_graph_index",
)

train_gqa_variant = dict(
    gqa_q_a=dict(**gqa_train_common_cfg, version="q-a", template_file=r"{{filedirname}}/template/VQA"),
    gqa_q_c=dict(**gqa_train_common_cfg, version="q-c", template_file=r"{{filedirname}}/template/VQA_CoT"),
    gqa_q_bc=dict(**gqa_train_common_cfg, version="q-bc", template_file=r"{{filedirname}}/template/VQA_BCoT"),
    gqa_q_s=dict(**gqa_train_common_cfg, version="q-s", template_file=r"{{filedirname}}/template/VQA_CoT"),
    gqa_q_bs=dict(**gqa_train_common_cfg, version="q-bs", template_file=r"{{filedirname}}/template/VQA_BCoT"),
    gqa_q_l=dict(**gqa_train_common_cfg, version="q-l", template_file=r"{{filedirname}}/template/VQA_CoT"),
    gqa_q_bl=dict(**gqa_train_common_cfg, version="q-bl", template_file=r"{{filedirname}}/template/VQA_BCoT"),

    gqa_qb_a=dict(**gqa_train_common_cfg, version="qb-a", template_file=r"{{filedirname}}/template/VQA"),
    gqa_qb_c=dict(**gqa_train_common_cfg, version="qb-c", template_file=r"{{filedirname}}/template/VQA_CoT"),
    gqa_qb_bc=dict(**gqa_train_common_cfg, version="qb-bc", template_file=r"{{filedirname}}/template/VQA_BCoT"),
    gqa_qb_s=dict(**gqa_train_common_cfg, version="qb-s", template_file=r"{{filedirname}}/template/VQA_CoT"),
    gqa_qb_bs=dict(**gqa_train_common_cfg, version="qb-bs", template_file=r"{{filedirname}}/template/VQA_BCoT"),
    gqa_qb_l=dict(**gqa_train_common_cfg, version="qb-l", template_file=r"{{filedirname}}/template/VQA_CoT"),
    gqa_qb_bl=dict(**gqa_train_common_cfg, version="qb-bl", template_file=r"{{filedirname}}/template/VQA_BCoT"),

    gqa_qbp_a=dict(**gqa_train_common_cfg, version="qbp-a", template_file=r"{{filedirname}}/template/VQA"),
    gqa_qbp_c=dict(**gqa_train_common_cfg, version="qbp-c", template_file=r"{{filedirname}}/template/VQA_CoT"),
    gqa_qbp_bc=dict(**gqa_train_common_cfg, version="qbp-bc", template_file=r"{{filedirname}}/template/VQA_BCoT"),
    gqa_qbp_s=dict(**gqa_train_common_cfg, version="qbp-s", template_file=r"{{filedirname}}/template/VQA_CoT"),
    gqa_qbp_bs=dict(**gqa_train_common_cfg, version="qbp-bs", template_file=r"{{filedirname}}/template/VQA_BCoT"),
    gqa_qbp_l=dict(**gqa_train_common_cfg, version="qbp-l", template_file=r"{{filedirname}}/template/VQA_CoT"),
    gqa_qbp_bl=dict(**gqa_train_common_cfg, version="qbp-bl", template_file=r"{{filedirname}}/template/VQA_BCoT"),
)

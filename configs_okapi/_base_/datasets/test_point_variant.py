point_test_common_cfg_local = dict(
    type='Point_QA_local',
    image_folder='zz1424:s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data',
    template_file=r"VQA",
)

point_test_common_cfg_twice = dict(
    type='Point_QA_twice',
    image_folder='zz1424:s3://publicdataset_8/Visual_Genome_Dataset_V1.2/unzip/data',
    template_file=r"VQA",
)

point_test_common_cfg_v7w = dict(
    type='V7W_POINT',
    image_folder='sh41:s3://MultiModal/Monolith/academic/v7w/data',
    template_file=r"VQA",
    do_shuffle_choice=True,
)

test_point_variant = dict(
    point_local_b_val=dict(**point_test_common_cfg_local, version='b', filename='{{fileDirname}}/../../../data/pointqa_local_val.jsonl'),
    point_local_p_val=dict(**point_test_common_cfg_local, version='p', filename='{{fileDirname}}/../../../data/pointqa_local_val.jsonl'),
    point_twice_oq_b_val=dict(**point_test_common_cfg_twice, version='oq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    point_twice_oq_p_val=dict(**point_test_common_cfg_twice, version='oq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    point_twice_sq_b_val=dict(**point_test_common_cfg_twice, version='sq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    point_twice_sq_p_val=dict(**point_test_common_cfg_twice, version='sq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    point_twice_gq_b_val=dict(**point_test_common_cfg_twice, version='gq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    point_twice_gq_p_val=dict(**point_test_common_cfg_twice, version='gq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_val.jsonl'),
    point_v7w_p_val=dict(**point_test_common_cfg_v7w, version='p', filename='{{fileDirname}}/../../../data/v7w_pointing_val.jsonl'),
    point_v7w_b_val=dict(**point_test_common_cfg_v7w, version='b', filename='{{fileDirname}}/../../../data/v7w_pointing_val.jsonl'),

    point_local_b_test=dict(**point_test_common_cfg_local, version='b', filename='{{fileDirname}}/../../../data/pointqa_local_test.jsonl'),
    point_local_p_test=dict(**point_test_common_cfg_local, version='p', filename='{{fileDirname}}/../../../data/pointqa_local_test.jsonl'),
    point_twice_oq_b_test=dict(**point_test_common_cfg_twice, version='oq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    point_twice_oq_p_test=dict(**point_test_common_cfg_twice, version='oq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    point_twice_sq_b_test=dict(**point_test_common_cfg_twice, version='sq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    point_twice_sq_p_test=dict(**point_test_common_cfg_twice, version='sq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    point_twice_gq_b_test=dict(**point_test_common_cfg_twice, version='gq-b', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    point_twice_gq_p_test=dict(**point_test_common_cfg_twice, version='gq-p', filename='{{fileDirname}}/../../../data/pointQA_twice_test.jsonl'),
    point_v7w_p_test=dict(**point_test_common_cfg_v7w, version='p', filename='{{fileDirname}}/../../../data/v7w_pointing_test.jsonl'),
    point_v7w_b_test=dict(**point_test_common_cfg_v7w, version='b', filename='{{fileDirname}}/../../../data/v7w_pointing_test.jsonl'),
)
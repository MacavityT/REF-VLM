point_test_common_cfg_local = dict(
    type='Point_QA_local',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual Genome',
    template_file=r"VQA",
)

point_test_common_cfg_twice = dict(
    type='Point_QA_twice',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual Genome',
    template_file=r"VQA",
)

point_test_common_cfg_v7w = dict(
    type='V7W_POINT',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual7W/images',
    template_file=r"VQA",
    do_shuffle_choice=True,
)

test_point_variant = dict(
    point_local_b_val=dict(**point_test_common_cfg_local, version='b', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_local_val.jsonl'),
    point_local_p_val=dict(**point_test_common_cfg_local, version='p', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_local_val.jsonl'),
    point_twice_oq_b_val=dict(**point_test_common_cfg_twice, version='oq-b', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_twice_oq_p_val=dict(**point_test_common_cfg_twice, version='oq-p', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_twice_sq_b_val=dict(**point_test_common_cfg_twice, version='sq-b', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_twice_sq_p_val=dict(**point_test_common_cfg_twice, version='sq-p', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_twice_gq_b_val=dict(**point_test_common_cfg_twice, version='gq-b', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_twice_gq_p_val=dict(**point_test_common_cfg_twice, version='gq-p', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_v7w_p_val=dict(**point_test_common_cfg_v7w, version='p', filename='/data/Aaronzhu/DatasetStage1/Shikra/v7w_pointing_val.jsonl'),
    point_v7w_b_val=dict(**point_test_common_cfg_v7w, version='b', filename='/data/Aaronzhu/DatasetStage1/Shikra/v7w_pointing_val.jsonl'),

    point_local_b_test=dict(**point_test_common_cfg_local, version='b', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_local_test.jsonl'),
    point_local_p_test=dict(**point_test_common_cfg_local, version='p', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_local_test.jsonl'),
    point_twice_oq_b_test=dict(**point_test_common_cfg_twice, version='oq-b', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_twice_oq_p_test=dict(**point_test_common_cfg_twice, version='oq-p', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_twice_sq_b_test=dict(**point_test_common_cfg_twice, version='sq-b', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_twice_sq_p_test=dict(**point_test_common_cfg_twice, version='sq-p', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_twice_gq_b_test=dict(**point_test_common_cfg_twice, version='gq-b', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_twice_gq_p_test=dict(**point_test_common_cfg_twice, version='gq-p', filename='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_v7w_p_test=dict(**point_test_common_cfg_v7w, version='p', filename='/data/Aaronzhu/DatasetStage1/Shikra/v7w_pointing_test.jsonl'),
    point_v7w_b_test=dict(**point_test_common_cfg_v7w, version='b', filename='/data/Aaronzhu/DatasetStage1/Shikra/v7w_pointing_test.jsonl'),
)
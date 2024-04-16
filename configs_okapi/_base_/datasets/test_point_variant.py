point_test_common_cfg_local = dict(
    type='Point_QA_local',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual Genome',
    template_name=r"VQA",
)

point_test_common_cfg_twice = dict(
    type='Point_QA_twice',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual Genome',
    template_name=r"VQA",
)

point_test_common_cfg_v7w = dict(
    type='V7W_POINT',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual7W/images',
    template_name=r"VQA",
    do_shuffle_choice=True,
)

test_point_variant = dict(
    point_local_b_val=dict(**point_test_common_cfg_local, version='b', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_local_val.jsonl'),
    point_local_p_val=dict(**point_test_common_cfg_local, version='p', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_local_val.jsonl'),
    point_twice_oq_b_val=dict(**point_test_common_cfg_twice, version='oq-b', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_twice_oq_p_val=dict(**point_test_common_cfg_twice, version='oq-p', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_twice_sq_b_val=dict(**point_test_common_cfg_twice, version='sq-b', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_twice_sq_p_val=dict(**point_test_common_cfg_twice, version='sq-p', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_twice_gq_b_val=dict(**point_test_common_cfg_twice, version='gq-b', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_twice_gq_p_val=dict(**point_test_common_cfg_twice, version='gq-p', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_val.jsonl'),
    point_v7w_p_val=dict(**point_test_common_cfg_v7w, version='p', text_path='/data/Aaronzhu/DatasetStage1/Shikra/v7w_pointing_val.jsonl'),
    point_v7w_b_val=dict(**point_test_common_cfg_v7w, version='b', text_path='/data/Aaronzhu/DatasetStage1/Shikra/v7w_pointing_val.jsonl'),

    point_local_b_test=dict(**point_test_common_cfg_local, version='b', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_local_test.jsonl'),
    point_local_p_test=dict(**point_test_common_cfg_local, version='p', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_local_test.jsonl'),
    point_twice_oq_b_test=dict(**point_test_common_cfg_twice, version='oq-b', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_twice_oq_p_test=dict(**point_test_common_cfg_twice, version='oq-p', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_twice_sq_b_test=dict(**point_test_common_cfg_twice, version='sq-b', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_twice_sq_p_test=dict(**point_test_common_cfg_twice, version='sq-p', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_twice_gq_b_test=dict(**point_test_common_cfg_twice, version='gq-b', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_twice_gq_p_test=dict(**point_test_common_cfg_twice, version='gq-p', text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_test.jsonl'),
    point_v7w_p_test=dict(**point_test_common_cfg_v7w, version='p', text_path='/data/Aaronzhu/DatasetStage1/Shikra/v7w_pointing_test.jsonl'),
    point_v7w_b_test=dict(**point_test_common_cfg_v7w, version='b', text_path='/data/Aaronzhu/DatasetStage1/Shikra/v7w_pointing_test.jsonl'),
)
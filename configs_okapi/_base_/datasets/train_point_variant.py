from xtuner.utils.constants import BOXES_PLACEHOLDER,POINTS_PLACEHOLDER

point_train_common_cfg_local = dict(
    type='Point_QA_local',
    text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_local_train.jsonl',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual Genome',
    image_info_folder='/data/Aaronzhu/DatasetStage1/Shikra/shape/vg100k_shape.jsonl',
    offline_processed_image_folder = '',
    template_name=r"VQA",
)

point_train_common_cfg_twice = dict(
    type='Point_QA_twice',
    text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_train.jsonl',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual Genome',
    image_info_folder='/data/Aaronzhu/DatasetStage1/Shikra/shape/vg100k_shape.jsonl',
    offline_processed_image_folder = '',
    template_name=r"VQA",
)

point_train_common_cfg_v7w = dict(
    type='V7W_POINT',
    text_path='/data/Aaronzhu/DatasetStage1/Shikra/v7w_pointing_train.jsonl',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual7W/images',
    image_info_folder='/data/Aaronzhu/DatasetStage1/Shikra/shape/visual7w_shape.jsonl',
    offline_processed_image_folder = '',
    template_name=r"VQA",
)

train_point_variant = dict(
    point_local_b=dict(
        **point_train_common_cfg_local,
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER],
        ), 
        version='b',
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_local_b',
    ),
    point_local_p=dict(
        **point_train_common_cfg_local, 
        version='p',
        map_placeholders=dict(
            input=[POINTS_PLACEHOLDER],
        ),
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_local_p',
    ),
    point_local_bp=dict(
        **point_train_common_cfg_local, 
        version='bp',
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER,POINTS_PLACEHOLDER],
        ), 
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_local_bp',
    ),
    point_twice_oq_b=dict(
        **point_train_common_cfg_twice, 
        version='oq-b',
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER],
        ), 
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_twice_oq_b',
    ),
    point_twice_oq_p=dict(
        **point_train_common_cfg_twice, 
        version='oq-p',
        map_placeholders=dict(
            input=[POINTS_PLACEHOLDER],
        ),
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_twice_oq_p',
    ),
    point_twice_oq_bp=dict(
        **point_train_common_cfg_twice, 
        version='oq-bp',
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER,POINTS_PLACEHOLDER],
        ), 
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_twice_oq_bp',
    ),
    point_twice_sq_b=dict(
        **point_train_common_cfg_twice, 
        version='sq-b',
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER],
        ), 
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_twice_sq_b',
    ),
    point_twice_sq_p=dict(
        **point_train_common_cfg_twice, 
        version='sq-p',
        map_placeholders=dict(
            input=[POINTS_PLACEHOLDER],
        ),
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_twice_sq_p',
    ),
    point_twice_sq_bp=dict(
        **point_train_common_cfg_twice, 
        version='sq-bp',
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER,POINTS_PLACEHOLDER],
        ), 
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_twice_sq_bp',
    ),
    point_twice_gq_b=dict(
        **point_train_common_cfg_twice, 
        version='gq-b',
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER],
        ), 
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_twice_gq_b',
    ),
    point_twice_gq_p=dict(
        **point_train_common_cfg_twice, 
        version='gq-p',
        map_placeholders=dict(
            input=[POINTS_PLACEHOLDER],
        ),
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_twice_gq_p',
    ),
    point_twice_gq_bp=dict(
        **point_train_common_cfg_twice, 
        version='gq-bp',
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER,POINTS_PLACEHOLDER],
        ), 
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_twice_gq_bp',
    ),
    point_v7w_p=dict(
        **point_train_common_cfg_v7w, 
        version='p',
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER],
            output=[POINTS_PLACEHOLDER],
        ),
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_v7w_p',
    ),
    point_v7w_b=dict(
        **point_train_common_cfg_v7w, 
        version='b',
        map_placeholders=dict(
            input=[BOXES_PLACEHOLDER],
            output=[BOXES_PLACEHOLDER],
        ),
        offline_processed_text_folder = '/data/Aaronzhu/DatasetStage1/offline_text_datasets/point_v7w_b',
    ),
)

# from itertools import product
# cls = ['POINT_LOCAL', 'POINT_TWICE', 'POINT_V7W']
# dfs = ['POINT_TRAIN_COMMON_CFG_LOCAL', 'POINT_TRAIN_COMMON_CFG_TWICE', 'POINT_TRAIN_COMMON_CFG_V7W']
# cfs = [
#     ['b', 'p', 'bp'],
#     list(map(lambda l: "-".join(l), (product(['oq', 'sq', 'gq'], ['b', 'p', 'bp'])))),
#     ['p', 'b'],
# ]
# for cl, df, cf in zip(cls, dfs, cfs):
#     for c in cf:
#         name = f"{cl}_{c.replace('-', '_')}"
#         print(f"{name}=dict(**{df}, version='{c}'),")

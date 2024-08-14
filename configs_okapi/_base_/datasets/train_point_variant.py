point_train_common_cfg_local = dict(
    type='Point_QA_local',
    text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_local_train.jsonl',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual Genome',
    template_name=r"VQA",
)

point_train_common_cfg_twice = dict(
    type='Point_QA_twice',
    text_path='/data/Aaronzhu/DatasetStage1/Shikra/pointQA_twice_train.jsonl',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual Genome',
    template_name=r"VQA",
)

point_train_common_cfg_v7w = dict(
    type='V7W_POINT',
    text_path='/data/Aaronzhu/DatasetStage1/Shikra/v7w_pointing_train.jsonl',
    image_folder='/data/Aaronzhu/DatasetStage1/Visual7W/images',
    template_name=r"VQA",
)

train_point_variant = dict(
    point_local_b=dict(
        **point_train_common_cfg_local,
        map_placeholders=dict(
            input=["<boxes>"],
        ), 
        version='b',
    ),
    point_local_p=dict(
        **point_train_common_cfg_local, 
        version='p',
        map_placeholders=dict(
            input=["<points>"],
        ),
    ),
    point_local_bp=dict(
        **point_train_common_cfg_local, 
        version='bp',
        map_placeholders=dict(
            input=["<boxes>","<points>"],
        ), 
    ),
    point_twice_oq_b=dict(
        **point_train_common_cfg_twice, 
        version='oq-b',
        map_placeholders=dict(
            input=["<boxes>"],
        ), 
    ),
    point_twice_oq_p=dict(
        **point_train_common_cfg_twice, 
        version='oq-p',
        map_placeholders=dict(
            input=["<points>"],
        ),
    ),
    point_twice_oq_bp=dict(
        **point_train_common_cfg_twice, 
        version='oq-bp',
        map_placeholders=dict(
            input=["<boxes>","<points>"],
        ), 
    ),
    point_twice_sq_b=dict(
        **point_train_common_cfg_twice, 
        version='sq-b',
        map_placeholders=dict(
            input=["<boxes>"],
        ), 
    ),
    point_twice_sq_p=dict(
        **point_train_common_cfg_twice, 
        version='sq-p',
        map_placeholders=dict(
            input=["<points>"],
        ),
    ),
    point_twice_sq_bp=dict(
        **point_train_common_cfg_twice, 
        version='sq-bp',
        map_placeholders=dict(
            input=["<boxes>","<points>"],
        ), 
    ),
    point_twice_gq_b=dict(
        **point_train_common_cfg_twice, 
        version='gq-b',
        map_placeholders=dict(
            input=["<boxes>"],
        ), 
    ),
    point_twice_gq_p=dict(
        **point_train_common_cfg_twice, 
        version='gq-p',
        map_placeholders=dict(
            input=["<points>"],
        ),
    ),
    point_twice_gq_bp=dict(
        **point_train_common_cfg_twice, 
        version='gq-bp',
        map_placeholders=dict(
            input=["<boxes>","<points>"],
        ), 
    ),
    point_v7w_p=dict(
        **point_train_common_cfg_v7w, 
        version='p',
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<points>"],
        ),
    ),
    point_v7w_b=dict(
        **point_train_common_cfg_v7w, 
        version='b',
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<boxes>"],
        ),
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

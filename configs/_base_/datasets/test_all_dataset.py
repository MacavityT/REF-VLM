from mmengine.config import read_base

with read_base():
    from test_rec_variant import test_rec_variant
    from test_flickr_variant import test_flickr_variant
    from test_gqa_variant import test_gqa_variant
    from test_clevr_variant import test_clevr_variant
    from test_gptgen_variant import test_gptgen_variant
    from test_vcr_variant import test_vcr_variant
    from test_vqav2_variant import test_vqav2_variant
    from test_point_variant import test_point_variant
    from test_pope_variant import test_pope_variant
    from test_interact_variant import test_interact_variant

test_all_dataset = dict(
    **test_rec_variant,
    **test_flickr_variant,
    **test_gqa_variant,
    **test_clevr_variant,
    **test_gptgen_variant,
    **test_vcr_variant,
    **test_vqav2_variant,
    **test_point_variant,
    **test_pope_variant,
    **test_interact_variant,
)

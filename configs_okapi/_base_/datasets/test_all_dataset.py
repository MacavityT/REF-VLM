_base_ = [
    'test_rec_variant.py',
    'test_flickr_variant.py',
    'test_gqa_variant.py',
    'test_clevr_variant.py',
    'test_gptgen_variant.py',
    'test_vcr_variant.py',
    'test_vqav2_variant.py',
    'test_point_variant.py',
    'test_pope_variant.py',
]

test_dataset = dict(
    **_base_.test_rec_variant,
    **_base_.test_flickr_variant,
    **_base_.test_gqa_variant,
    **_base_.test_clevr_variant,
    **_base_.test_gptgen_variant,
    **_base_.test_vcr_variant,
    **_base_.test_vqav2_variant,
    **_base_.test_point_variant,
    **_base_.test_pope_variant,
)

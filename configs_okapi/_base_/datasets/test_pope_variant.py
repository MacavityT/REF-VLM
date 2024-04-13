pope_test_common_cfg = dict(
    type='POPEVQADataset',
    image_folder=r'openmmlab1424:s3://openmmlab/datasets/detection/coco/val2014',
)

test_pope_variant = dict(
    coco_pope_random_q_a=dict(
        **pope_test_common_cfg,
        filename='{{fileDirname}}/../../../data/coco_pope_random.jsonl',
        template_file=r'VQA'
    ),
    coco_pope_random_q_bca=dict(
        **pope_test_common_cfg,
        filename='{{fileDirname}}/../../../data/coco_pope_random.jsonl',
        template_file=r'VQA_BCoT'
    ),
    coco_pope_popular_q_a=dict(
        **pope_test_common_cfg,
        filename='{{fileDirname}}/../../../data/coco_pope_popular.jsonl',
        template_file=r'VQA'
    ),
    coco_pope_popular_q_bca=dict(
        **pope_test_common_cfg,
        filename='{{fileDirname}}/../../../data/coco_pope_popular.jsonl',
        template_file=r'VQA_BCoT'
    ),
    coco_pope_adversarial_q_a=dict(
        **pope_test_common_cfg,
        filename='{{fileDirname}}/../../../data/coco_pope_adversarial.jsonl',
        template_file=r'VQA'
    ),
    coco_pope_adversarial_q_bca=dict(
        **pope_test_common_cfg,
        filename='{{fileDirname}}/../../../data/coco_pope_adversarial.jsonl',
        template_file=r'VQA_BCoT'
    ),
)
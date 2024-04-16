pope_test_common_cfg = dict(
    type='POPEVQADataset',
    image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/val',
)

test_pope_variant = dict(
    coco_pope_random_q_a=dict(
        **pope_test_common_cfg,
        text_path='/data/Aaronzhu/DatasetStage1/Shikra/coco_pope_random.jsonl',
        template_name=r'VQA'
    ),
    coco_pope_random_q_bca=dict(
        **pope_test_common_cfg,
        text_path='/data/Aaronzhu/DatasetStage1/Shikra/coco_pope_random.jsonl',
        template_name=r'VQA_BCoT'
    ),
    coco_pope_popular_q_a=dict(
        **pope_test_common_cfg,
        text_path='/data/Aaronzhu/DatasetStage1/Shikra/coco_pope_popular.jsonl',
        template_name=r'VQA'
    ),
    coco_pope_popular_q_bca=dict(
        **pope_test_common_cfg,
        text_path='/data/Aaronzhu/DatasetStage1/Shikra/coco_pope_popular.jsonl',
        template_name=r'VQA_BCoT'
    ),
    coco_pope_adversarial_q_a=dict(
        **pope_test_common_cfg,
        text_path='/data/Aaronzhu/DatasetStage1/Shikra/coco_pope_adversarial.jsonl',
        template_name=r'VQA'
    ),
    coco_pope_adversarial_q_bca=dict(
        **pope_test_common_cfg,
        text_path='/data/Aaronzhu/DatasetStage1/Shikra/coco_pope_adversarial.jsonl',
        template_name=r'VQA_BCoT'
    ),
)
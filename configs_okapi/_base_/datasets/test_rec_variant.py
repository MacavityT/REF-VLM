rec_test_common_cfg = dict(
    type='RECDataset',
    template_file=r'REC',
    image_folder=r'/data/Aaronzhu/DatasetStage1/MSCOCO/2014/train',
    max_dynamic_size=None,
)

test_rec_variant = dict(
    rec_refcocog_umd_test=dict(
        **rec_test_common_cfg,
        filename=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_refcocog_umd_test.jsonl',
    ),
    rec_refcocoa_unc_testa=dict(
        **rec_test_common_cfg,
        filename=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_refcoco+_unc_testA.jsonl',
    ),
    rec_refcocoa_unc_testb=dict(
        **rec_test_common_cfg,
        filename=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_refcoco+_unc_testB.jsonl',
    ),
    rec_refcoco_unc_testa=dict(
        **rec_test_common_cfg,
        filename=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_refcoco_unc_testA.jsonl',
    ),
    rec_refcoco_unc_testb=dict(
        **rec_test_common_cfg,
        filename=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_refcoco_unc_testB.jsonl',
    ),
    rec_refcocog_umd_val=dict(
        **rec_test_common_cfg,
        filename=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_refcocog_umd_val.jsonl',
    ),
    rec_refcocoa_unc_val=dict(
        **rec_test_common_cfg,
        filename=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_refcoco+_unc_val.jsonl',
    ),
    rec_refcoco_unc_val=dict(
        **rec_test_common_cfg,
        filename=r'/data/Aaronzhu/DatasetStage1/Shikra/REC_refcoco_unc_val.jsonl',
    ),
)

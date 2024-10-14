vcr_val_common_cfg = dict(
    type='VCRDataset',
    image_folder=r'/data/Aaronzhu/DatasetStage1/vcr1/vcr1images',
    text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/vcr_val.jsonl',
)

vcr_test_common_cfg = dict(
    type='VCRPredDataset',
    image_folder=r'/data/Aaronzhu/DatasetStage1/vcr1/vcr1images',
    text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/vcr_test.jsonl',
)

test_vcr_variant = dict(
    vcr_val_q_ra=dict(**vcr_val_common_cfg, version='q-ra', template_name=r"VQA_BCoT", ),
    vcr_val_qc_a=dict(**vcr_val_common_cfg, version='qc-a', template_name=r"VQA", ),
    vcr_val_qc_rac=dict(**vcr_val_common_cfg, version='qc-rac', template_name=r"VQA_BCoT", ),
    vcr_val_qac_r=dict(**vcr_val_common_cfg, version='qac-r', template_name=r"VQA", ),
    vcr_val_qc_a_qc_r=dict(**vcr_val_common_cfg, version='qc-a-qc-r', template_name=r"VQA", ),

    vcr_test_qc_a=dict(**vcr_test_common_cfg, version='qc-a', template_name=r"VQA", ),
    vcr_test_qc_rac=dict(**vcr_test_common_cfg, version='qc-rac', template_name=r"VQA_BCoT", ),
    vcr_test_qac_r=dict(**vcr_test_common_cfg, version='qac-r', template_name=r"VQA", ),
    vcr_test_qc_a_qc_r=dict(**vcr_test_common_cfg, version='qc-a-qc-r', template_name=r"VQA", ),
)

# ccfg = 'VCR_TEST_COMMON_CFG'
# splits = [
#     'val',
#     'test',
# ]
# versions = [
#     'qc-a', 'qc-ra', 'qc-rac',  # for evaluation: A
#     'qac-r', 'qc-a-qc-r',  # for evaluation: R
# ]
# for split in splits:
#     for v in versions:
#         name = f"VCR_{split}_{v.replace('-', '_')}"
#         text_path = fr'{{fileDirname}}/../../../data/vcr_{split}.jsonl'
#         print(f"{name}=dict(**{ccfg}, version='{v}', text_path=r'{text_path}'),")

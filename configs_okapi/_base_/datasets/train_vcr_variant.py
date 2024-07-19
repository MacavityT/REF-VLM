vcr_train_common_cfg = dict(
    type='VCRDataset',
    text_path=r'/data/Aaronzhu/DatasetStage1/Shikra/vcr_train.jsonl',
    image_folder=r'/data/Aaronzhu/DatasetStage1/vcr1/vcr1images',
)

train_vcr_variant = dict(
    vcr_q_a=dict(
        **vcr_train_common_cfg, 
        version='q-a', 
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<boxes>"],
        ),
        template_name=r"VQA",
        # offline_processed_text_folder='/data/Aaronzhu/DatasetStage1/offline_text_datasets/vcr_q_a',
    ),
    vcr_q_ra=dict(
        **vcr_train_common_cfg, 
        version='q-ra', 
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<boxes>"],
        ),
        template_name=r"VQA_BCoT",
        # offline_processed_text_folder='/data/Aaronzhu/DatasetStage1/offline_text_datasets/vcr_q_ra'
    ),
    vcr_qc_a=dict(
        **vcr_train_common_cfg, 
        version='qc-a', 
        map_placeholders=dict(
            input=["<boxes>"],
        ),
        template_name=r"VQA",
        # offline_processed_text_folder='/data/Aaronzhu/DatasetStage1/offline_text_datasets/vcr_qc_a'
    ),
    vcr_qc_ra=dict(
        **vcr_train_common_cfg, 
        version='qc-ra', 
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<boxes>"],
        ),
        template_name=r"VQA_BCoT",
        # offline_processed_text_folder='/data/Aaronzhu/DatasetStage1/offline_text_datasets/vcr_qc_ra'
    ),
    vcr_qc_rac=dict(
        **vcr_train_common_cfg, 
        version='qc-rac', 
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<boxes>"],
        ),
        template_name=r"VQA_BCoT",
        # offline_processed_text_folder='/data/Aaronzhu/DatasetStage1/offline_text_datasets/vcr_qc_rac'
    ),
    vcr_qa_r=dict(
        **vcr_train_common_cfg, 
        version='qa-r', 
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<boxes>"],
        ),
        template_name=r"VQA",
        # offline_processed_text_folder='/data/Aaronzhu/DatasetStage1/offline_text_datasets/vcr_qa_r'
    ),
    vcr_q_a_q_r=dict(
        **vcr_train_common_cfg, 
        version='q-a-q-r', 
        map_placeholders=dict(
            input=["<boxes>"],
            output=["<boxes>"],
        ),
        template_name=r"VQA",
        # offline_processed_text_folder='/data/Aaronzhu/DatasetStage1/offline_text_datasets/vcr_q_a_q_r'
    ),
    vcr_qac_r=dict(
        **vcr_train_common_cfg, 
        version='qac-r', 
        map_placeholders=dict(
            input=["<boxes>"],
        ),
        template_name=r"VQA",
        # offline_processed_text_folder='/data/Aaronzhu/DatasetStage1/offline_text_datasets/vcr_qac_r'
    ),
    vcr_qc_a_qc_r=dict(
        **vcr_train_common_cfg, 
        version='qc-a-qc-r', 
        map_placeholders=dict(
            input=["<boxes>"],
        ),
        template_name=r"VQA",
        # offline_processed_text_folder='/data/Aaronzhu/DatasetStage1/offline_text_datasets/vcr_qc_a_qc_r'
    ),
)


# ccfg = 'vcr_train_common_cfg'
# versions = [
#     'q-a', 'q-ra',
#     'qc-a', 'qc-ra', 'qc-rac',  # for evaluation: A
#     'qa-r', 'q-a-q-r',
#     'qac-r', 'qc-a-qc-r',  # for evaluation: R
# ]
# for v in versions:
#     name = f"VCR_{v.replace('-', '_')}"
#     print(f"{name}=dict(**{ccfg}, version='{v}'),")

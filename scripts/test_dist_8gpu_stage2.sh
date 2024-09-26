

gpu_num=0
checkpoint="/model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0923_full_512_sft_box_det/iter_16420.pth"

NPOC_PER_NODE=$gpu_num xtuner test /code/okapi-mllm/configs_okapi/okapi_7b_test_stage2.py --checkpoint $checkpoint
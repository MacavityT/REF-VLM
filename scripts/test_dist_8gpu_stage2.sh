

gpu_num=6
checkpoint="/model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0530/iter_1000.pth"

NPOC_PER_NODE=$gpu_num xtuner test /code/okapi-mllm/configs_okapi/okapi_7b_test_stage2.py --checkpoint $checkpoint
#!/bin/bash
# --resume /model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0718/iter_7000.pth
NPROC_PER_NODE=8 xtuner train configs_okapi/okapi_7b_train_stage2.py --deepspeed deepspeed_zero2 --work-dir /model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0718 --resume /model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0718/iter_8500.pth 
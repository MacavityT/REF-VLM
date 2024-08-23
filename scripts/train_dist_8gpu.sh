#!/bin/bash

# NPROC_PER_NODE=8 xtuner train configs_okapi/okapi_7b_train_stage2_decoder.py --deepspeed deepspeed_zero2 --work-dir /model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0813 --resume /model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0813/iter_23500.pth
NPROC_PER_NODE=8 xtuner train configs_okapi/okapi_7b_train_stage2_decoder.py --deepspeed deepspeed_zero2 --work-dir /model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0822mask
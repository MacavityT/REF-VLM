#!/bin/bash

NPROC_PER_NODE=8 xtuner train configs_okapi/okapi_7b_train_stage2_keypoint.py --deepspeed deepspeed_zero2 --work-dir /model/Aaronzhu/OkapiModel/vicuna_7b/stage2/1016_keypoint
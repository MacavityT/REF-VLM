#!/bin/bash
# NPROC_PER_NODE=8 xtuner train configs_okapi/sketch_exp/debug.py --deepspeed deepspeed_zero2
NPROC_PER_NODE=4 xtuner train configs_okapi/okapi_7b_train_stage2_keypoint.py --deepspeed deepspeed_zero2 \
	--work-dir work_dirs/debug
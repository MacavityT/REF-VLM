#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
# NPROC_PER_NODE=8 xtuner train configs_okapi/sketch_exp/debug.py --deepspeed deepspeed_zero2
NPROC_PER_NODE=4 xtuner train configs/train_stage3_keypoint.py --deepspeed deepspeed_zero2 \
	--work-dir work_dirs/debug
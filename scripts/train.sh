#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)


NPROC_PER_NODE=8 xtuner train configs/train_stage3_keypoint.py  --deepspeed deepspeed_zero2 \
	--work-dir ./work_dirs/kpt_decoder/0302_kpt
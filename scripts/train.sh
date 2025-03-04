#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)


NPROC_PER_NODE=8 xtuner train configs/ablation/train_stage2_vpt.py  --deepspeed deepspeed_zero2 \
	--work-dir ./work_dirs/ablation/0304_ours_no_mask_token
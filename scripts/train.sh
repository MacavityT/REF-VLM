#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)


NPROC_PER_NODE=8 xtuner train configs/ablation/train_stage2_det.py  --deepspeed deepspeed_zero2 \
	--work-dir ./work_dirs/ablation/0305_det_no_match
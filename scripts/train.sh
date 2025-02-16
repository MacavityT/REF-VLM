#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)


NPROC_PER_NODE=8 xtuner train configs/train_stage2_vd_adapter.py  --deepspeed deepspeed_zero2 \
	--work-dir /code/VT-PLUG/checkpoints/vicuna_7b/stage2_ref/1128_ref_sam
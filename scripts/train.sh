#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

NPROC_PER_NODE=8 xtuner train configs/train_stage3_sam.py --deepspeed deepspeed_zero2 \
	--work-dir /model/Aaronzhu/OkapiModel/vicuna_7b/finetune/1119_sam_res
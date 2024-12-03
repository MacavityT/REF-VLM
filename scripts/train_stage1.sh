#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)


NPROC_PER_NODE=8 xtuner train configs/train_stage1.py  --deepspeed deepspeed_zero2 \
	--work-dir /code/VT-PLUG/checkpoints/Andychen/Stage_1/1203_stage1
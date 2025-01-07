#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)


NPROC_PER_NODE=8 xtuner train configs/train_stage2_cn.py  --deepspeed deepspeed_zero2 \
	--work-dir /model/Aaronzhu/OkapiModel/Qwen2.5/xtuner_output/stage2/0107
#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

NPROC_PER_NODE=8 xtuner train configs/train_stage2_nodecoder.py --deepspeed deepspeed_zero2 \
	--work-dir /model/Aaronzhu/OkapiModel/vicuna_7b/stage2_ref/1112_llm_unfreeze
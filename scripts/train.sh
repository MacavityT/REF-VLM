#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

NPROC_PER_NODE=8 xtuner train configs/train_stage2_decoder.py --deepspeed deepspeed_zero2 \
	--work-dir /model/Aaronzhu/OkapiModel/vicuna_7b/finetune/1117_nocot  --resume checkpoints/vicuna_7b/finetune/1117_nocot/iter_1500.pth
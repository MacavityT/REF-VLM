#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)

NPROC_PER_NODE=8 xtuner train configs/train_stage2_nodecoder.py --deepspeed deepspeed_zero2   --work-dir /checkpoints/Andychen/1108_unfreeze_3358
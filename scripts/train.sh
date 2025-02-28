#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)


NPROC_PER_NODE=8 xtuner train configs/train_stage2_vd_adapter.py  --deepspeed deepspeed_zero2 \
	--work-dir ./work_dirs/0222_det_freeze_llm_norefshift
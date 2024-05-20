#!/bin/bash

conda activate next_gpt 
NPROC_PER_NODE=8 xtuner train configs_okapi/okapi_7b_train_stage2.py --deepspeed deepspeed_zero2
#!/bin/bash

conda activate next_gpt 
echo $VC_TASK1_HOSTS | awk -F, '{print $1}'
echo $VC_TASK1_HOSTS | awk -F, '{print $2}'
echo $VC_TASK1_HOSTS | awk -F, '{print $3}'
echo $VC_TASK1_HOSTS | awk -F, '{print $4}'
echo $VC_TASK1_HOSTS | awk -F, '{print $5}'



host1_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $1}')
host2_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $2}')
host3_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $3}')
host4_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $4}')
host5_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $5}')


local_host=$(hostname)
echo "local host: $local_host"


host1=$(echo $host1_addr | cut -d \. -f 1)
if test $local_host = $host1
then
    echo "host1: $host1"
    NPROC_PER_NODE=8 NNODES=5 ADDR=$host1_addr NODE_RANK=0 PORT=29505 xtuner train configs_okapi/sketch_full_stage_1_online.py --deepspeed deepspeed_zero2 --work-dir /output
fi


host2=$(echo $host2_addr | cut -d \. -f 1)
if test $local_host = $host2
then
    echo "host2: $host2"
    NPROC_PER_NODE=8 NNODES=5 ADDR=$host1_addr NODE_RANK=1 PORT=29505 xtuner train configs_okapi/sketch_full_stage_1_online.py --deepspeed deepspeed_zero2  --work-dir /output
fi

host3=$(echo $host3_addr | cut -d \. -f 1)
if test $local_host = $host3
then
    echo "host3: $host3"
    NPROC_PER_NODE=8 NNODES=5 ADDR=$host1_addr NODE_RANK=2 PORT=29505 xtuner train configs_okapi/sketch_full_stage_1_online.py --deepspeed deepspeed_zero2  --work-dir /output
fi

host4=$(echo $host4_addr | cut -d \. -f 1)
if test $local_host = $host4
then
    echo "host4: $host4"
    NPROC_PER_NODE=8 NNODES=5 ADDR=$host1_addr NODE_RANK=3 PORT=29505 xtuner train configs_okapi/sketch_full_stage_1_online.py --deepspeed deepspeed_zero2  --work-dir /output
fi

host5=$(echo $host5_addr | cut -d \. -f 1)
if test $local_host = $host5
then
    echo "host5: $host5"
    NPROC_PER_NODE=8 NNODES=5 ADDR=$host1_addr NODE_RANK=4 PORT=29505 xtuner train configs_okapi/sketch_full_stage_1_online.py --deepspeed deepspeed_zero2  --work-dir /output
fi




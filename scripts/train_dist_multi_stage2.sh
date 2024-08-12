#!/bin/bash


echo $VC_TASK1_HOSTS | awk -F, '{print $1}'
echo $VC_TASK1_HOSTS | awk -F, '{print $2}'
echo $VC_TASK1_HOSTS | awk -F, '{print $3}'
echo $VC_TASK1_HOSTS | awk -F, '{print $4}'
echo $VC_TASK1_HOSTS | awk -F, '{print $5}'
echo $VC_TASK1_HOSTS | awk -F, '{print $6}'
echo $VC_TASK1_HOSTS | awk -F, '{print $7}'


host1_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $1}')
host2_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $2}')
host3_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $3}')
host4_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $4}')
host5_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $5}')
host6_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $6}')
host7_addr=$(echo $VC_TASK1_HOSTS | awk -F, '{print $7}')

work_dir="/model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0715"
checkpoint="/model/Aaronzhu/OkapiModel/vicuna_7b/stage2/0607/iter_7000.pth"
local_host=$(hostname)
node=7
echo "local host: $local_host"

host1=$(echo $host1_addr | cut -d \. -f 1)
if test $local_host = $host1
then
    echo "host1: $host1"
    NPROC_PER_NODE=8 NNODES=$node ADDR=$host1_addr NODE_RANK=0 PORT=29505 xtuner train configs_okapi/okapi_7b_train_stage2.py --deepspeed deepspeed_zero2 --work-dir $work_dir 
fi

host2=$(echo $host2_addr | cut -d \. -f 1)
if test $local_host = $host2
then
    echo "host2: $host2"
    NPROC_PER_NODE=8 NNODES=$node ADDR=$host1_addr NODE_RANK=1 PORT=29505 xtuner train configs_okapi/okapi_7b_train_stage2.py --deepspeed deepspeed_zero2  --work-dir $work_dir
fi

host3=$(echo $host3_addr | cut -d \. -f 1)
if test $local_host = $host3
then
    echo "host3: $host3"
    NPROC_PER_NODE=8 NNODES=$node ADDR=$host1_addr NODE_RANK=2 PORT=29505 xtuner train configs_okapi/okapi_7b_train_stage2.py --deepspeed deepspeed_zero2  --work-dir $work_dir
fi

host4=$(echo $host4_addr | cut -d \. -f 1)
if test $local_host = $host4
then
    echo "host4: $host4"
    NPROC_PER_NODE=8 NNODES=$node ADDR=$host1_addr NODE_RANK=3 PORT=29505 xtuner train configs_okapi/okapi_7b_train_stage2.py --deepspeed deepspeed_zero2  --work-dir $work_dir
fi

host5=$(echo $host5_addr | cut -d \. -f 1)
if test $local_host = $host5
then
    echo "host5: $host5"
    NPROC_PER_NODE=8 NNODES=$node ADDR=$host1_addr NODE_RANK=4 PORT=29505 xtuner train configs_okapi/okapi_7b_train_stage2.py --deepspeed deepspeed_zero2  --work-dir $work_dir
fi

host6=$(echo $host6_addr | cut -d \. -f 1)
if test $local_host = $host6
then
    echo "host6: $host6"
    NPROC_PER_NODE=8 NNODES=$node ADDR=$host1_addr NODE_RANK=5 PORT=29505 xtuner train configs_okapi/okapi_7b_train_stage2.py --deepspeed deepspeed_zero2  --work-dir $work_dir
fi

host7=$(echo $host7_addr | cut -d \. -f 1)
if test $local_host = $host7
then
    echo "host7: $host7"
    NPROC_PER_NODE=8 NNODES=$node ADDR=$host1_addr NODE_RANK=6 PORT=29505 xtuner train configs_okapi/okapi_7b_train_stage2.py --deepspeed deepspeed_zero2  --work-dir $work_dir
fi

# host8=$(echo $host6_addr | cut -d \. -f 1)
# if test $local_host = $host8
# then
#     echo "host8: $host8"
#     NPROC_PER_NODE=8 NNODES=$node ADDR=$host1_addr NODE_RANK=7 PORT=29505 xtuner train configs_okapi/okapi_7b_train_stage2.py --deepspeed deepspeed_zero2  --work-dir $work_dir
# fi

#!/bin/bash

conda activate next_gpt 
echo $VC_TASK1_HOSTS | awk -F, '{print $1}'
echo $VC_TASK1_HOSTS | awk -F, '{print $2}'
echo $VC_TASK1_HOSTS | awk -F, '{print $3}'
echo $VC_TASK1_HOSTS | awk -F, '{print $4}'
echo $VC_TASK1_HOSTS | awk -F, '{print $5}'



host1=$(echo $VC_TASK1_HOSTS | awk -F, '{print $1}')
host2=$(echo $VC_TASK1_HOSTS | awk -F, '{print $2}')
host3=$(echo $VC_TASK1_HOSTS | awk -F, '{print $3}')
host4=$(echo $VC_TASK1_HOSTS | awk -F, '{print $4}')
host5=$(echo $VC_TASK1_HOSTS | awk -F, '{print $5}')




local_host=$(hostname)
echo "local host: $local_host"


echo "$host1 slots=8" > scripts/myhostfile.txt
echo "$host2 slots=8" >> scripts/myhostfile.txt
echo "$host3 slots=8" >> scripts/myhostfile.txt
echo "$host4 slots=8" >> scripts/myhostfile.txt
echo "$host5 slots=8" >> scripts/myhostfile.txt



host1=$(echo $host1 | cut -d \. -f 1)
echo "host1: $host1"

if test $local_host = $host1
then
    NPROC_PER_NODE=8 xtuner train configs_okapi/okapi_7b_train_stage1.py --deepspeed deepspeed_zero2  --hostfile scripts/myhostfile.txt   
fi

sleep 30d

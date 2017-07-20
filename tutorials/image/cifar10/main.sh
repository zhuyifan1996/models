#! /bin/bash

NRUNS=$1
for i in `seq 1 $NRUNS`
do
    time -p python cifar10_multi_gpu_train.py --num_gpus=4
done

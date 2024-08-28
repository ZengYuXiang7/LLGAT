#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量

# CPU Experiment
train_sizes="900"
ranks="60 80 100 120 140 160 180 200 300 400 500"
for train_size in $train_sizes; do
   for rank in $ranks; do
       python train_model.py --config_path ./exper_config.py --exp_name TestConfig --train_size "$train_size" --rank "$rank"
   done
done

train_sizes="900"
ranks="60 80 100 120 140 160 180 200 300 400 500"
for train_size in $train_sizes; do
   for rank in $ranks; do
       python train_model.py --config_path ./exper_config.py --exp_name TestGPUConfig --train_size "$train_size" --rank "$rank"
   done
done


################################
#orders="1 2 3 4 5"
#for order in $orders; do
#   for train_size in $train_sizes; do
#       python train_model.py --config_path ./exper_config.py --exp_name TestConfig --train_size "$train_size" --order "$order"
#   done
#done


#orders="1 2 3 4 5"
#for train_size in $train_sizes; do
#   for order in $orders; do
#       python train_model.py --config_path ./exper_config.py --exp_name TestGPUConfig --train_size "$train_size" --order "$order"
#   done
#done






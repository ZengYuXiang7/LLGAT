#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量

 LLMs Ablation Experiment
 train_sizes="100 200 400 500 900"
 for train_size in $train_sizes; do
     python train_model.py --config_path ./exper_config.py --exp_name TestConfig --train_size $train_size --llm 0
 done

 for train_size in $train_sizes; do
     python train_model.py --config_path ./exper_config.py --exp_name TestConfig --train_size $train_size --op_encoder value
     python train_model.py --config_path ./exper_config.py --exp_name TestConfig --train_size $train_size --op_encoder one_hot
 done

 for train_size in $train_sizes; do
     python train_model.py --config_path ./exper_config.py --exp_name TestConfig --train_size $train_size --graph_encoder gcn
     python train_model.py --config_path ./exper_config.py --exp_name TestConfig --train_size $train_size --graph_encoder graphsage
 done

# GPU Experiment
train_sizes="100 200 400 500 900"
for train_size in $train_sizes; do
    python train_model.py --config_path ./exper_config.py --exp_name TestConfig --dataset gpu --train_size "$train_size" --llm 0
done

for train_size in $train_sizes; do
    python train_model.py --config_path ./exper_config.py --exp_name TestConfig --dataset gpu --train_size "$train_size" --op_encoder value
    python train_model.py --config_path ./exper_config.py --exp_name TestConfig --dataset gpu --train_size "$train_size" --op_encoder one_hot
done

for train_size in $train_sizes; do
    python train_model.py --config_path ./exper_config.py --exp_name TestConfig --dataset gpu --train_size "$train_size" --graph_encoder gcn
    python train_model.py --config_path ./exper_config.py --exp_name TestConfig --dataset gpu --train_size "$train_size" --graph_encoder graphsage
done


#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量

train_sizes="400"
for train_size in $train_sizes; do
# #    python train_model.py --config_path ./exper_config.py --exp_name BrpNASConfig --train_size $train_size
# #    python train_model.py --config_path ./exper_config.py --exp_name MLPConfig --train_size $train_size
# #    python train_model.py --config_path ./exper_config.py --exp_name LSTMConfig --train_size $train_size
# #    python train_model.py --config_path ./exper_config.py --exp_name GRUConfig --train_size $train_size
# #    python train_model.py --config_path ./exper_config.py --exp_name BiRnnConfig --train_size $train_size
# #    python train_model.py --config_path ./exper_config.py --exp_name GCNConfig --train_size $train_size
#     python train_model.py --config_path ./exper_config.py --exp_name TestConfig --train_size $train_size
     python train_model.py --config_path ./exper_config.py --exp_name TestGPUConfig --train_size $train_size
done




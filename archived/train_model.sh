#!/bin/bash
clear
ulimit -s unlimited
ulimit -a
# 定义变量

#train_sizes="50 100 200 500 900"
train_sizes="200"
for train_size in $train_sizes; do
    python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --train_size $train_size --epochs 700 --patience 150 --rounds 5 --att 0 --cross 0
#    python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --train_size $train_size --epochs 700 --patience 150 --rounds 5 --att 1 --cross 0 --heads 4
#    python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --train_size $train_size --epochs 700 --patience 150 --rounds 5 --att 1 --cross 0 --heads 6

#    python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --train_size $train_size --epochs 700 --patience 150 --rounds 5 --att 1 --cross 1 --heads 4
#    python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --train_size $train_size --epochs 700 --patience 150 --rounds 5 --att 1 --cross 1 --heads 6

done
#python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 900 --llm 1 \
#       --device_name i7-7820x --train_device desktop-cpu-core-i7-7820x-fp32 --eval_device desktop-cpu-core-i9-13900k-fp32


#python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 900 --llm 1 \
#       --device_name i9-13900k --train_device desktop-cpu-core-i9-13900k-fp32 --eval_device desktop-cpu-core-i7-7820x-fp32

#python eval_inductive.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 900 --llm 1 \
#       --device_name i7-7820x --train_device desktop-cpu-core-i7-7820x-fp32

################################################################################################################
#python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 3600 --llm 1 \
#       --device_name i9-13900k --train_device desktop-cpu-core-i9-13900k-fp32 --eval_device desktop-cpu-core-i7-7820x-fp32 --rounds 1

#python meta_inductive.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 3600 --llm 1 \
#       --device_name i7-7820x --train_device desktop-cpu-core-i7-7820x-fp32 --lr 0.001
################################################################################################################


###############################################################################################################
#python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 3600 --llm 1 \
#       --device_name i7-12700h --train_device desktop-cpu-core-i7-12700h-fp32 --eval_device desktop-cpu-core-i7-7820x-fp32 --rounds 1
#
#python meta_inductive.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 3600 --llm 1 \
#       --device_name i7-7820x --train_device desktop-cpu-core-i7-7820x-fp32 --lr 0.001
###############################################################################################################

###############################################################################################################
#python train_model.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 3600 --llm 1 \
#       --device_name i7-12700h --train_device desktop-cpu-core-i7-12700h-fp32 --eval_device  --rounds 1

#python eval_inductive.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 3600 --llm 1 \
#       --device_name i7-7820x --train_device desktop-cpu-core-i7-7820x-fp32

#python meta_inductive.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 3600 --llm 1 \
#       --device_name i7-12700h --train_device desktop-cpu-core-i7-7820x-fp32 --lr 0.001
#
#python meta_inductive.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 3600 --llm 1 \
#       --device_name i7-7820x --train_device desktop-cpu-core-i7-7820x-fp32 --lr 0.001

#python meta_inductive.py --config_path ./exper_config.py --exp_name LLMConfig --train_size 3600 --llm 1 \
#       --device_name i9-13900k --train_device desktop-cpu-core-i9-13900k-fp32 --lr 0.001
###############################################################################################################


#python train_model.py --config_path ./exper_config.py --exp_name BrpNASConfig --train_size 3600 --llm 1 \
#       --device_name i7-12700h --train_device desktop-cpu-core-i7-12700h-fp32 --eval_device  --rounds 1


#python meta_inductive.py --config_path ./exper_config.py --exp_name BrpNASConfig --train_size 3600 --llm 1 \
#       --device_name i7-7820x --train_device desktop-cpu-core-i7-7820x-fp32 --lr 0.001
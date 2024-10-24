# coding : utf-8
# Author : yuxiang Zeng
import os
import subprocess
import time
from datetime import datetime
import sys
sys.dont_write_bytecode = True

def debug(commands):
    commands.append(f"python train_model.py --config_path ./exper_config.py --exp_name MLPConfig "
                       f"--train_size 100 --retrain 1 --dataset cpu --rank 300")
    return commands

def Baselines(commands):
    train_sizes = [100, 200, 400, 500, 900]

    train_sizes = [100]
    exps = ['MLPConfig', 'BrpNASConfig', 'LSTMConfig', 'GRUConfig', 'BiRnnConfig', 'FlopsConfig', 'DNNPerfConfig']
    exps = ['MLPConfig']
    for train_size in train_sizes:
        for exp in exps:
            command = (f"python train_model.py --config_path ./exper_config.py --exp_name {exp} "
                       f"--train_size {train_size} --retrain 1 --dataset cpu --rank 300")
            commands.append(command)
    train_sizes = [100, 200, 400, 500, 900]
    for train_size in train_sizes:
        for exp in exps:
            command = (f"python train_model.py --config_path ./exper_config.py --exp_name {exp} "
                       f"--train_size {train_size} --retrain 0 --dataset gpu --rank 300")
            commands.append(command)
    return commands


def Ablation(commands):
    # train_sizes = [100, 200, 400, 500, 900]
    train_sizes = [50]
    rank = 100
    for ablation in [6]:
        encoder = 'one_hot' if ablation in [3, 5, 6] else 'embed'
        for train_size in train_sizes:
            command = (f"python train_model.py --config_path ./exper_config.py --exp_name TestConfig "
                       f"--train_size {train_size} --retrain 0 --rank {rank} --Ablation {ablation} --op_encoder {encoder} --debug 1")
            commands.append(command)
        for train_size in train_sizes:
            command = (f"python train_model.py --config_path ./exper_config.py --exp_name TestGPUConfig "
                       f"--train_size {train_size} --retrain 0 --rank {rank} --Ablation {ablation} --op_encoder {encoder} --debug 1")
            commands.append(command)
    return commands


def Our_model(commands):
    # train_sizes = [100, 200, 400, 500, 900]
    train_sizes = [100]
    ranks = [100]
    for rank in ranks:
        for train_size in train_sizes:
            command = (f"python train_model.py --config_path ./exper_config.py --exp_name TestConfig "
                       f"--train_size {train_size} --retrain 0 --rank {rank}")
            commands.append(command)
    for rank in ranks:
        for train_size in train_sizes:
            command = (f"python train_model.py --config_path ./exper_config.py --exp_name TestGPUConfig "
                       f"--train_size {train_size} --retrain 0 --rank {rank}")
            commands.append(command)
    return commands


def experiment_command():
    commands = []
    commands = Our_model(commands)
    return commands


def run_command(command, log_file, retry_count=0):
    success = False
    while not success:
        # 获取当前时间并格式化
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

        # 如果是重试的命令，标记为 "Retrying"
        if retry_count > 0:
            retry_message = "Retrying"
        else:
            retry_message = "Running"

        # 将执行的命令和时间写入日志文件
        with open(log_file, 'a') as f:
            f.write(f"{retry_message} at {current_time}: {command}\n")

        # 直接执行命令，将输出和错误信息打印到终端
        process = subprocess.run(command, shell=True)

        # 根据返回码判断命令是否成功执行
        if process.returncode == 0:
            success = True
        else:
            with open(log_file, 'a') as f:
                f.write(f"Command failed, retrying in 3 seconds: {command}\n")
            retry_count += 1
            time.sleep(3)  # 等待一段时间后重试


def main():
    log_file = "run.log"

    # 清空日志文件的内容
    with open(log_file, 'a') as f:
        f.write(f"Experiment Start!!!\n")

    commands = experiment_command()

    # 执行所有命令
    for command in commands:
        run_command(command, log_file)

    with open(log_file, 'a') as f:
        f.write(f"All commands executed successfully.\n")


if __name__ == "__main__":
    main()

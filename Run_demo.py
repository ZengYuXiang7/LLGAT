# coding : utf-8
# Author : yuxiang Zeng
import os
import subprocess
import time
from datetime import datetime
import sys
sys.dont_write_bytecode = True

def experiment_command():
    # train_sizes = [100, 200, 400, 500, 900]
    # ranks = [100]
    # 创建一个命令列表
    commands = []

    # # 添加 CPU Experiment 的命令
    # for rank in ranks:
    #     for train_size in train_sizes:
    #         command = f"python train_model.py --config_path ./exper_config.py --exp_name TestConfig --train_size {train_size} --retrain 0 --rank {rank}"
    #         commands.append(command)
    #
    # # 添加 GPU Experiment 的命令
    # for rank in ranks:
    #     for train_size in train_sizes:
    #         command = f"python train_model.py --config_path ./exper_config.py --exp_name TestGPUConfig --train_size {train_size} --retrain 0 --rank {rank}"
    #         commands.append(command)


    # 添加 Baselines 的命令
    train_sizes = [100]
    # exps = ['MLPConfig', 'BrpNASConfig', 'LSTMConfig', 'GRUConfig', 'BiRnnConfig', 'FlopsConfig', 'DNNPerfConfig']
    exps = ['MLPConfig']
    for train_size in train_sizes:
        for exp in exps:
            command = f"python train_model.py --config_path ./exper_config.py --exp_name {exp} --train_size {train_size} --retrain 0 --dataset cpu --rank 300"
            commands.append(command)
            
    # train_sizes = [100, 200, 400, 500, 900]
    # for train_size in train_sizes:
    #     for exp in exps:
    #         command = f"python train_model.py --config_path ./exper_config.py --exp_name {exp} --train_size {train_size} --retrain 0 --dataset gpu --rank 300"
    #         commands.append(command)

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
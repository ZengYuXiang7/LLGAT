# coding : utf-8
# Author : yuxiang Zeng
import os
import subprocess
import time
from datetime import datetime
import sys
sys.dont_write_bytecode = True


def experiment_command():
    train_sizes = [900]
    ranks = [68, 80, 100, 120, 140, 160, 180, 200, 300, 400, 500]
    # 创建一个命令列表，每个元素包含命令和执行状态
    commands = []

    # 添加 CPU Experiment 的命令
    for rank in ranks:
        for train_size in train_sizes:
            command = f"python train_model.py --config_path ./exper_config.py --exp_name TestConfig --train_size {train_size} --retrain 0 --rank {rank}"
            commands.append((command, False))

    # 添加 GPU Experiment 的命令
    for rank in ranks:
        for train_size in train_sizes:
            command = f"python train_model.py --config_path ./exper_config.py --exp_name TestGPUConfig --train_size {train_size} --retrain 0 --rank {rank}"
            commands.append((command, False))

    return commands


def run_command(command, log_file, retry=False):
    # 获取当前时间并格式化
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 如果是重试的命令，标记为 "Retrying"
    retry_message = "Retrying" if retry else "Running"

    # 将执行的命令和时间写入日志文件
    with open(log_file, 'a') as f:
        f.write(f"{retry_message} at {current_time}: {command}\n")

    # 直接执行命令，将输出和错误信息打印到终端
    process = subprocess.run(command, shell=True)

    # 根据返回码判断命令是否成功执行
    return process.returncode == 0



def main():
    log_file = "run.log"

    # 清空日志文件的内容
    with open(log_file, 'a') as f:
        f.write(f"Experiment Start!!!\n")

    commands = experiment_command()

    # 执行所有命令并更新状态
    all_succeeded = False
    while not all_succeeded:
        all_succeeded = True
        for i, (command, success) in enumerate(commands):
            if not success:  # 只重新执行失败的命令
                retry = not all_succeeded  # 标记为重试
                success = run_command(command, log_file, retry)
                commands[i] = (command, success)  # 更新执行状态
                if not success:
                    all_succeeded = False  # 还有失败的命令，需要继续重试

        if not all_succeeded:
            with open(log_file, 'a') as f:
                f.write(f"Some commands failed, retrying in 10 seconds...\n")
            time.sleep(10)  # 等待一段时间后重试

    with open(log_file, 'a') as f:
        f.write(f"All commands executed successfully.\n")


if __name__ == "__main__":
    main()
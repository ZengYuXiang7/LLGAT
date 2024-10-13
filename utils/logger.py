# coding : utf-8
# Author : yuxiang Zeng
# 日志
import logging
import pickle
import sys
import time
import numpy as np
import platform
import glob
import os

from utils.utils import makedir

class Logger:
    def __init__(self, filename, exper_datail, args):
        self.args = args
        self.clear_the_useless_logs()
        self.exper_datail = exper_datail
        self.fileroot = f'./results/{args.model}/' + time.strftime('%Y%m%d', time.localtime(time.time())) + '/log/'
        self.filename = filename
        makedir(self.fileroot)
        exper_time = time.strftime('%H_%M_%S', time.localtime(time.time())) + '_'
        self.exper_filename = self.fileroot + exper_time + self.filename
        logging.basicConfig(level=logging.INFO, filename=f"{self.exper_filename}.log", filemode='w')
        self.logger = logging.getLogger(self.args.model)
        self.log(self.format_and_sort_args_dict(args.__dict__))

    def save_result(self, metrics):
        args = self.args
        makedir('./results/metrics/')
        address = f'./results/metrics/' + self.filename
        for key in metrics:
            pickle.dump(np.mean(metrics[key]), open(address + key + 'mean.pkl', 'wb'))
            pickle.dump(np.std(metrics[key]), open(address + key + 'std.pkl', 'wb'))

    # 日志记录
    def log(self, string):
        import time
        if string[0] == '\n':
            print('\n', end='')
            string = string[1:]
        final_string = time.strftime('|%Y-%m-%d %H:%M:%S| ', time.localtime(time.time())) + string
        self.only_print(string)
        self.logger.info(final_string)

    def __call__(self, string):
        self.log(string)

    def only_print(self, string):
        import time
        if string[0] == '\n':
            print('\n', end='')
            string = string[1:]
        green_string = f"\033[1;38;2;151;200;129m{time.strftime('|%Y-%m-%d %H:%M:%S| ', time.localtime(time.time()))}\033[0m" + f"\033[1m{string}\033[0m"
        print(green_string)

    def show_epoch_error(self, runId, epoch, monitor, epoch_loss, result_error, train_time):
        if self.args.verbose and epoch % self.args.verbose == 0:
            self.only_print(f"\033[1;38;2;151;200;129m{self.exper_datail}\033[0m")
            self.only_print(f'Best NMAE Epoch {monitor.best_epoch} = {monitor.best_score * -1:.4f}  now = {(epoch - monitor.best_epoch):d}')
            if self.args.classification:
                self.only_print(f"Round={runId + 1} Epoch={epoch + 1:03d} Loss={epoch_loss:.4f} vAcc={result_error['Acc']:.4f} vF1={result_error['F1']:.4f} vPrecision={result_error['P']:.4f} vRecall={result_error['Recall']:.4f} time={sum(train_time):.1f} s ")
            else:
                self.only_print(f"Round={runId + 1} Epoch={epoch + 1:03d} Loss={epoch_loss:.4f} vMAE={result_error['MAE']:.4f} vRMSE={result_error['RMSE']:.4f} vNMAE={result_error['NMAE']:.4f} vNRMSE={result_error['NRMSE']:.4f} time={sum(train_time):.1f} s ")
                # self.only_print(f"Acc = [1%={result_error['Acc_1']:.4f}, 5%={result_error['Acc_5']:.4f}, 10%={result_error['Acc_10']:.4f}] ")

    def show_test_error(self, runId, monitor, results, sum_time):
        if self.args.classification:
            self(f'Round={runId + 1} BestEpoch={monitor.best_epoch:3d} Acc={results["Acc"]:.4f} F1={results["F1"]:.4f} Precision={results["P"]:.4f} Recall={results["Recall"]:.4f} Training_time={sum_time:.1f} s \n')
        else:
            self(f'Round={runId + 1} BestEpoch={monitor.best_epoch:3d} MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f} Training_time={sum_time:.1f} s ')
            # self(f"Acc = [1%={results['Acc_1']:.4f}, 5%={results['Acc_5']:.4f}, 10%={results['Acc_10']:.4f}] ")
        print()

    def format_and_sort_args_dict(self, args_dict, items_per_line=4):
        # Sort the dictionary by keys
        sorted_items = sorted(args_dict.items())
        formatted_str = ''
        for i in range(0, len(sorted_items), items_per_line):
            line_items = sorted_items[i:i + items_per_line]
            line_str = ', '.join([f"'{key}': {value}" for key, value in line_items])
            formatted_str += f"     {line_str},\n"
        return f"{{\n{formatted_str}}}".strip(',\n')


    def clear_the_useless_logs(self):
        def delete_small_log_files(directory):
            # 获取所有.log文件
            log_files = glob.glob(os.path.join(directory, '*.log'))
            number_lines = 23

            # 遍历所有的.log文件
            for file_path in log_files:
                try:
                    # 检查文件是否存在且可读
                    if not os.path.isfile(file_path) or not os.access(file_path, os.R_OK):
                        print(f"Cannot access '{file_path}'. Skipping.")
                        continue

                    # 使用with语句安全地打开文件
                    with open(file_path, 'r') as file:
                        lines = file.readlines()

                    # 检查行数并删除文件
                    if len(lines) < number_lines:
                        os.remove(file_path)  # 删除文件
                        print(f"Deleted '{file_path}' as it had less than {number_lines} lines.")

                except OSError as e:
                    print(f"OS error processing file {file_path}: {e}")
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

        def delete_empty_directories(dir_path):
            import os
            # 检查目录是否存在
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                # 遍历目录中的所有文件和子目录，从最底层开始
                for root, dirs, files in os.walk(dir_path, topdown=False):
                    # 先删除空的子目录
                    for name in dirs:
                        dir_to_remove = os.path.join(root, name)
                        # 如果目录是空的，则删除它
                        try:
                            if not os.listdir(dir_to_remove):  # 判断目录是否为空
                                os.rmdir(dir_to_remove)
                                print(f"Directory {dir_to_remove} has been deleted.")
                        except FileNotFoundError:
                            # 如果目录已经不存在，忽略此错误
                            pass
                    # 检查当前目录是否也是空的，如果是则删除它
                    try:
                        if not os.listdir(root):  # 判断当前根目录是否为空
                            os.rmdir(root)
                            print(f"Directory {root} has been deleted.")
                    except FileNotFoundError:
                        # 如果目录已经不存在，忽略此错误
                        pass
            else:
                print(f"Directory {dir_path} does not exist.")

        # 使用os.walk来遍历目录
        root_directory = f'./results/'
        delete_empty_directories(root_directory)
        for dirpath, dirnames, filenames in os.walk(root_directory):
            if 'log' in dirpath:
                delete_small_log_files(dirpath)

    # 邮件发送日志
    def send_email(self, subject, body, receiver_email="zengyuxiang@hnu.edu.cn"):
        import yagmail
        import pickle
        import os

        if isinstance(body, dict):  # 判断 body 是否是字典
            metrics = body
            body = []
            # 添加实验结果部分
            body.append('*' * 10 + 'Experiment Results:' + '*' * 10)
            for key in metrics:
                body.append(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')

            # 添加实验成功提示
            body.append('*' * 10 + 'Experiment Success' + '*' * 10)

            for i in range(self.args.rounds):
                round_metrics = f"Round {i + 1} : "
                for key in metrics:
                    round_metrics += f"{key}: {metrics[key][i]:.4f} "
                body.append(round_metrics)

            # 添加格式化的 args 字典信息
            body.append(self.format_and_sort_args_dict(self.args.__dict__, 3))
        else:
            print(body)
            temp = body
            body = [self.format_and_sort_args_dict(self.args.__dict__, 3), temp]

        # 获得主目录下的邮件授权码，这部分为个人私有。
        email_code_address = os.path.expanduser('~') + '/qq_smtp_info.pickle'
        try:
            with open(email_code_address, 'rb') as f:
                all_info = pickle.load(f)
            sender_email = all_info['email']
            sender_password = all_info['password']
        except FileNotFoundError:
            print("非管理员，无法发送邮件")
            return False

        try:
            # 发送 HTML 格式的邮件
            yag = yagmail.SMTP(user=sender_email, password=sender_password, host='smtp.qq.com')
            yag.send(to=receiver_email, subject=subject, contents=body)
            print("邮件发送成功!")
        except Exception as e:
            print(f"发送邮件时出错: {e}")


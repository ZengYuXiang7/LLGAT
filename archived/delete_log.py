import os
import glob


def delete_small_log_files(directory):
    # 设置文件路径
    log_files = glob.glob(os.path.join(directory, '*.log'))
    number_lines = 19
    # 遍历所有的.log文件
    for file_path in log_files:
        try:
            # 打开并读取文件
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # 检查行数
            if len(lines) < number_lines:
                os.remove(file_path)  # 删除文件
                print(f"Deleted '{file_path}' as it had less than {number_lines} lines.")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

def clear_the_useless_logs():
    # 指定要扫描的目录，如 'path/to/your/log-directory'
    for i in range(1, 13):
        for j in range(1, 32):
            for item in ['ours', 'birnn', 'brp_nas', 'gcn', 'gru', 'lstm', 'mlp']:
                delete_small_log_files(f'results/{item}/2024{i:02d}{j:02d}/log')

if __name__ == '__main__':

    clear_the_useless_logs()
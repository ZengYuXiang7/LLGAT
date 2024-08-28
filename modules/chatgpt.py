import os
import pickle

from utils.config import get_config
from utils.utils import get_file_name
import openai

openai.api_base = 'https://api.openai-proxy.org/v1'
openai.api_key = 'sk-IMnRtlj5qFJCjd9WYh9yGLZnWv7r8mn0O7anskka7Ve4Q2XX@29805'

class ChatGPT:
    def __init__(self, args):
        self.args = args
        # self.model = "gpt-4"
        self.model = "gpt-3.5-turbo"

    def chat_once(self, user_input):
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": user_input},
            ],
            temperature=0.7,  # 调整随机性
            max_tokens=150,  # 输出的最大长度
            frequency_penalty=0.5,  # 减少重复
            presence_penalty=0.5  # 鼓励创新
        )
        model_reply = response['choices'][0]['message']['content']
        return model_reply



class NAS_ChatGPT:
    def __init__(self, args):
        self.args = args
        self.gpt = ChatGPT(args)

    def get_device_more_info(self, input_device, device_type='cpu'):
        if device_type == 'cpu':
            prompt = self.contructing_cpu_promote(input_device)
        else:
            prompt = self.contructing_gpu_promote(input_device)
        model_reply = self.gpt.chat_once(prompt)
        return model_reply

    def contructing_cpu_promote(self, input_device):
        """
            CPU
            CPU时钟频率::最大睿频频率::核心数::线程数::三级缓存::最大内存带宽
            CPU Clock Frequency, Maximum Turbo Boost Frequency, Number of Cores, Number of Threads, Level 3 Cache, Maximum Memory Bandwidth
            CPU_Clock_Frequency::Maximum_Turbo_Boost_Frequency::Number_of_Cores::Number_of_Threads::Level_3_Cache::Maximum_Memory_Bandwidth
        """
        pre_str = "You are now a search engines, and required to provide the inquired information of the given CPU processer.\n"
        input_device = 'The processer is ' + input_device + '.\n'
        output_format = "The inquired information is : Maximum Turbo Boost Frequency, Number of Cores, Number of Threads, Level 3 Cache, Maximum Memory Bandwidth.\n \
                        And please output them in form of: [Maximum_Turbo_Boost_Frequency, Number_of_Cores, Number_of_Threads, Level_3_Cache, Maximum_Memory_Bandwidth]. \n  \
                        please output only the content in the form above, i.e., [%.2lf, %d, %d, %.2lf, %.2lf]\n, \
                        only number, but no other thing else, no unit symbol, no reasoning, no index.\n\n"
        prompt = pre_str + input_device + output_format
        return prompt

    def contructing_gpu_promote(self, input_device):
        """
            GPU
            流处理器数量::核芯频率::显存::位宽
            Stream processor count, Core clock frequency, Video memory, Memory bus width
            Stream_processor_count::Core_clock_frequency::Video_memory::Memory_bus_width
        """
        pre_str = "You are now a search engines, and required to provide the inquired information of the given GPU processer.\n"
        input_device = 'The processer is ' + input_device + '.\n'
        output_format = "The inquired information is : Stream processor count, Core clock frequency, Video memory, Memory bus width\n \
                         And please output them in form of: [Stream_processor_count, Core_clock_frequency, Video_memory, Memory_bus_width], \
                         please output only the content in the form above, i.e., [%.2lf, %.2lf, %.2lf, %.2lf]\n, \
                         only number, but no other thing else, no unit symbol, no reasoning, no index.\n\n"
        prompt = pre_str + input_device + output_format
        return prompt

def excute(device_name, dtype):
    str_list = llm.get_device_more_info(device_name, dtype)
    import ast
    list_obj = ast.literal_eval(str_list)
    list_obj = list(list_obj)
    os.makedirs('./agu', exist_ok=True)
    # 将列表写入文件
    with open(f'./agu/{device_name}.pkl', 'wb') as f:
        pickle.dump(list_obj, f)
    print("list : ", list_obj)
    print('Done!')


if __name__ == '__main__':
    args = 1
    # NAS llm
    llm = NAS_ChatGPT(args)
    # device_name = 'core-i7-7820x'
    device_name = 'core-i9-13900K'
    # excute('core-i7-7820x', 'cpu')
    # excute('core-i9-13900K', 'cpu')
    excute('1080Ti', 'gpu')
    excute('3080', 'gpu')



# coding : utf-8
# Author : yuxiang Zeng


# First, install torchprofile
# !pip install torchprofile
import torch
import torchprofile

from train_model import Model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# def estimate_gpu_memory(model, sample_input, args):
#     # Assuming input_size is a tuple, e.g., (batch_size, channels, height, width)
#     model = model.to(args.device)
#     dummy_input = tuple(item.to(args.device) for item in sample_input)
#     with torch.cuda.profiler.profile():
#         model(dummy_input)
#     memory_allocated = torch.cuda.memory_allocated()
#     memory_reserved = torch.cuda.memory_reserved()
#     print(f"Memory Allocated: {memory_allocated / (1024 ** 2):.2f} MB")
#     print(f"Memory Reserved: {memory_reserved / (1024 ** 2):.2f} MB")
#     torch.cuda.empty_cache()


def calculate_flops(model, sample_input, args):
    from thop import profile
    # 计算 FLOPs 和参数数量
    graph, features, value = tuple(item.to(args.device) for item in sample_input)
    flops, params = profile(model, inputs=sample_input[:-1])
    print(f"Flops: {flops}, Params: {params}")


if __name__ == '__main__':
    # Experiment Settings, logger, plotter
    from utils.config import get_config
    from utils.logger import Logger
    from utils.plotter import MetricsPlotter
    from utils.utils import set_settings, set_seed

    args = get_config()
    set_settings(args)
    log_filename = f'Model_{args.model}_{args.dataset}_S{args.train_size}_R{args.rank}_Ablation{args.Ablation}'
    plotter = MetricsPlotter(log_filename, args)
    log = Logger(log_filename, plotter, args)

    from data import experiment, DataModule

    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(datamodule, args)

    ###########################################################################
    sample_inputs = next(iter(datamodule.train_loader))
    total_params = count_parameters(model)
    print(f"Total Trainable Parameters: {total_params}")

    calculate_flops(model, sample_inputs, args)
    estimate_gpu_memory(model, sample_inputs, args)

    # Inference time
    pass
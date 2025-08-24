# coding : utf-8
# Author : Anonymous

from default_config import *
from dataclasses import dataclass


@dataclass
class TestConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 100
    # Encoder
    op_encoder: str = 'embed'
    # GAT
    graph_encoder: str = 'gat'
    order: int = 4
    heads: int = 8
    # LLMs
    llm: int = 1

    # Dataset
    dataset: str = 'cpu'
    device_name: str = 'core-i7-7820x'  # 'core-i9-13900k'
    train_device: str = 'desktop-cpu-core-i7-7820x-fp32'
    eval_device: str = 'desktop-cpu-core-i9-13900k-fp32'


@dataclass
class TestGPUConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 100

    # Encoder
    op_encoder: str = 'embed'
    # GAT
    graph_encoder: str = 'gat'
    order: int = 3
    heads: int = 8
    # LLMs
    llm: int = 1

    # Dataset
    dataset: str = 'gpu'
    device_name: str = '1080Ti'  # 'core-i9-13900k'
    train_device: str = 'desktop-gpu-gtx-1080ti-fp32'


@dataclass
class HELPConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'help'
    rank: int = 300


@dataclass
class MLPConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp'
    rank: int = 300


@dataclass
class GCNConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'gcn'
    rank: int = 300


@dataclass
class BrpNASConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'brp_nas'
    bs: int = 1
    rank: int = 300


@dataclass
class LSTMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'lstm'
    rank: int = 300

@dataclass
class GRUConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'gru'
    rank: int = 300

@dataclass
class BiRnnConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'birnn'
    rank: int = 300


@dataclass
class FlopsConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'flops'
    rank: int = 32

@dataclass
class DNNPerfConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'dnnperf'
    lr: float = 0.0001
    bs: int = 64
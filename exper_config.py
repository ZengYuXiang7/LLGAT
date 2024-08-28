# coding : utf-8
# Author : yuxiang Zeng

from default_config import *
from dataclasses import dataclass


@dataclass
class TestConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 160

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

    att: int = 0
    cross: int = 0


@dataclass
class TestGPUConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'ours'
    bs: int = 32
    rank: int = 160

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

    att: int = 0
    cross: int = 0


@dataclass
class HELPConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'help'


@dataclass
class MLPConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'mlp'


@dataclass
class GCNConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'gcn'
    rank: int = 300


@dataclass
class BrpNASConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'brp_nas'
    bs: int = 1


@dataclass
class LSTMConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'lstm'

@dataclass
class GRUConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'gru'

@dataclass
class BiRnnConfig(ExperimentConfig, BaseModelConfig, LoggerConfig, DatasetInfo, TrainingConfig, OtherConfig):
    model: str = 'birnn'




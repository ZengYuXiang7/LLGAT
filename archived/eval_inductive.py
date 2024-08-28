# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import *
from baselines.birnn import BiRNN
from baselines.brp_nas import BRP_NAS
from baselines.gru import GRU
from baselines.help import HELPBase
from baselines.lstm import LSTM
from baselines.mlp import MLP
from data import experiment, DataModule
from baselines.gnn import GraphSAGEConv
from modules.gnn_llm import OurModel
from modules.inductive import InductiveModel
from data import experiment, DataModule, TensorDataset, get_dataloaders, custom_collate_fn
from modules.gnn_llm import OurModel
from utils.config import get_config
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.plotter import MetricsPlotter
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import set_settings, set_seed
global log, args

torch.set_default_dtype(torch.float32)

import pysnooper

# @pysnooper.snoop()


class NewDeviceData:
    def __init__(self, exper, args):
        self.args = args
        raw_data = exper.load_data(args)
        self.x, self.y = exper.preprocess_data(raw_data, args)
        self.max_value = np.max(self.y)
        self.y /= self.max_value
        print(self.x.shape, self.y.shape)
        self.test_set = TensorDataset(self.x, self.y, args)
        self.max_workers = 3
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=256,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, args),
            num_workers=self.max_workers,
        )

class Model(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.args = args
        self.input_size = data.x.shape[-1]
        self.hidden_size = args.rank
        if args.model == 'ours':
            # self.model = InductiveModel(args)
            self.model = OurModel(args)

        elif args.model == 'brp_nas':
            self.model = BRP_NAS(args)

        elif args.model == 'gcn':
            self.model = GraphSAGEConv(6, args.rank, 6, self.args)

        elif args.model == 'mlp':
            self.model = MLP(6, self.hidden_size, 1, args)

        elif args.model == 'lstm':
            self.model = LSTM(6, self.hidden_size, 1, args)

        elif args.model == 'gru':
            self.model = GRU(6, self.hidden_size, 1, args)

        elif args.model == 'birnn':
            self.model = BiRNN(6, self.hidden_size, 1, args)

        elif args.model == 'help':
            self.model = HELPBase(6, args)

        else:
            raise ValueError(f"Unsupported model type: {args.model}")

    def forward(self, adjacency, features):
        y = self.model.forward(adjacency, features)
        return y.flatten()

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        val_loss = 0.
        preds = []
        reals = []
        dataloader = dataModule.valid_loader if mode == 'valid' else dataModule.test_loader
        for batch in tqdm(dataloader):
            graph, features, value = batch
            graph, features, value = graph.to(self.args.device), features.to(self.args.device), value.to(self.args.device)
            pred = self.forward(graph, features)
            if mode == 'valid':
                val_loss += self.loss_function(pred, value)
            if self.args.classification:
                pred = torch.max(pred, 1)[1]  # 获取预测的类别标签
            preds.append(pred)
            reals.append(value)
        reals = torch.cat(reals, dim=0)
        preds = torch.cat(preds, dim=0)
        if mode == 'valid':
            self.scheduler.step(val_loss)
        metrics_error = ErrorMetrics(reals * dataModule.max_value, preds * dataModule.max_value, self.args)
        return metrics_error

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min' if args.classification else 'max', factor=0.5, patience=args.patience // 1.5, threshold=0.0)


def RunExperiments(log, args):
    # Set seed
    set_seed(args.seed)
    exper = experiment(args)
    datamodule = NewDeviceData(exper, args)
    model = Model(datamodule, args)
    model_path = f'./checkpoints/{args.model}/{log_filename}_round_{0}.pt'
    model.setup_optimizer(args)
    model.load_state_dict(torch.load(model_path))
    results = model.evaluate_one_epoch(datamodule, 'test')
    log.only_print(f'MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f}')
    log.only_print(f"Acc = [1%={results['Acc_1']:.4f}, 5%={results['Acc_5']:.4f}, 10%={results['Acc_10']:.4f}] ")
    # return results


if __name__ == '__main__':
    args = get_config()
    set_settings(args)
    args.inductive = True

    # logger plotter
    exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model}, Train_size : {args.train_size}, Bs : {args.bs}, Rank : {args.rank}"
    log_filename = f'Model_{args.model}_Trainsize{args.train_size}_Rank{args.rank}'
    log = Logger(log_filename, exper_detail, args)
    plotter = MetricsPlotter(log_filename, args)
    args.log = log
    log(str(args.__dict__))

    # Run Experiment
    RunExperiments(log, args)


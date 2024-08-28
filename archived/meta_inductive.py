# coding : utf-8
# Author : yuxiang Zeng
import collections
import os.path
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
from utils.monitor import EarlyStopping
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
        with open(os.path.join(args.path, args.dataset, args.train_device) + '.pickle', 'rb') as f:
            import pickle
            df = pickle.load(f)
            self.max_value = 0
            for key in df.keys():
                self.max_value = max(self.max_value, df[key])
        # self.max_value = np.max(self.y)
        self.y /= self.max_value
        train_size = 100
        valid_size = int(len(self.x) * 0.05)

        self.train_x = self.x[:train_size]
        self.train_y = self.y[:train_size]

        self.valid_x = self.x[train_size:train_size + valid_size]
        self.valid_y = self.y[train_size:train_size + valid_size]

        self.test_x = self.x[train_size + valid_size:]
        self.test_y = self.y[train_size + valid_size:]
        print(len(self.train_x), len(self.valid_x), len(self.test_x))
        self.train_set = TensorDataset(self.train_x, self.train_y, args)
        self.valid_set = TensorDataset(self.valid_x, self.valid_y, args)
        self.test_set = TensorDataset(self.test_x, self.test_y, args)

        self.max_workers = 3
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=args.bs,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, args),
            num_workers=self.max_workers,
        )
        self.valid_loader = DataLoader(
            self.valid_set,
            batch_size=256,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, args),
            num_workers=self.max_workers,
        )
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=256,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, args),
            num_workers=self.max_workers,
        )
        args.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)} Max_value : {self.max_value:.2f}')


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

    def train_one_epoch(self, dataModule):
        loss = None
        self.train()
        torch.set_grad_enabled(True)
        t1 = time.time()
        for train_Batch in (dataModule.train_loader):
            graph, features, value = train_Batch
            graph, features, value = graph.to(self.args.device), features.to(self.args.device), value.to(self.args.device)
            pred = self.forward(graph, features)
            loss = self.loss_function(pred, value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        return loss, t2 - t1

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        val_loss = 0.
        preds = []
        reals = []
        dataloader = dataModule.valid_loader if mode == 'valid' else dataModule.test_loader
        for batch in (dataloader):
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
    model.load_state_dict(torch.load(model_path))
    monitor = EarlyStopping(args)

    # Start Finetune
    model.setup_optimizer(args)
    train_time = []
    for epoch in trange(args.epochs):
        epoch_loss, time_cost = model.train_one_epoch(datamodule)
        valid_error = model.evaluate_one_epoch(datamodule, 'valid')
        monitor.track_one_epoch(epoch, model, valid_error)
        train_time.append(time_cost)
        log.show_epoch_error(0, epoch, monitor, epoch_loss, valid_error, train_time)
        plotter.append_epochs(valid_error)
        if monitor.early_stop:
            break
    model.load_state_dict(monitor.best_model)
    sum_time = sum(train_time[: monitor.best_epoch])
    results = model.evaluate_one_epoch(datamodule, 'test')
    log.show_test_error(0, monitor, results, sum_time)


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


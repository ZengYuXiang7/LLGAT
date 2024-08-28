# coding : utf-8
# Author : yuxiang Zeng
import collections
import time
import os
import numpy as np
import pandas as pd
import torch

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
from utils.config import get_config
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.plotter import MetricsPlotter
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed
from utils.utils import makedir
global log, args
import sys
sys.dont_write_bytecode = True
torch.set_default_dtype(torch.float32)

import pysnooper

# @pysnooper.snoop()
class Model(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.args = args
        self.input_size = data.x.shape[-1]
        self.hidden_size = args.rank
        if args.model == 'ours':
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

    def setup_optimizer(self, args):
        self.to(args.device)
        self.loss_function = get_loss_function(args).to(args.device)
        self.optimizer = get_optimizer(self.parameters(), lr=args.lr, decay=args.decay, args=args)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min' if args.classification else 'max', factor=0.5, patience=args.patience // 1.5, threshold=0.0)

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


def RunOnce(args, runId, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(datamodule, args)
    monitor = EarlyStopping(args)
    makedir(f'./checkpoints/{args.model}')
    model_path = f'./checkpoints/{args.model}/{log_filename}_round_{runId}.pt'

    # Check if retrain is required or if model file exists
    retrain_required = args.retrain == 1 or not os.path.exists(model_path)

    if not retrain_required:
        try:
            model.setup_optimizer(args)
            model.load_state_dict(torch.load(model_path))
            results = model.evaluate_one_epoch(datamodule, 'test')
            if not args.classification:
                log(f'MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f}')
            else:
                log(f'Acc={results["Acc"]:.4f} F1={results["F1"]:.4f} Precision={results["P"]:.4f} Recall={results["Recall"]:.4f}')
            args.record = False
        except Exception as e:
            log.only_print(f'Error: {str(e)}')
            retrain_required = True

    if retrain_required:
        # Setup training tool
        model.setup_optimizer(args)
        train_time = []
        for epoch in trange(args.epochs):
            epoch_loss, time_cost = model.train_one_epoch(datamodule)
            valid_error = model.evaluate_one_epoch(datamodule, 'valid')
            monitor.track_one_epoch(epoch, model, valid_error)
            train_time.append(time_cost)
            log.show_epoch_error(runId, epoch, monitor, epoch_loss, valid_error, train_time)
            plotter.append_epochs(valid_error)
            if monitor.early_stop:
                break
        model.load_state_dict(monitor.best_model)
        sum_time = sum(train_time[: monitor.best_epoch])
        results = model.evaluate_one_epoch(datamodule, 'test')
        log.show_test_error(runId, monitor, results, sum_time)
        # Save the best model parameters
        torch.save(monitor.best_model, model_path)
        log.only_print(f'Model parameters saved to {model_path}')

    return results


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)

    for runId in range(args.rounds):
        plotter.reset_round()
        try:
            results = RunOnce(args, runId, log)
            plotter.append_round()
            for key in results:
                metrics[key].append(results[key])
        except Exception as e:
            log(f'Run {runId + 1} Error: {e}, This run will be skipped.')

    log('*' * 20 + 'Experiment Results:' + '*' * 20)

    for key in metrics:
        log(f'{key}: {np.mean(metrics[key]):.4f} ± {np.std(metrics[key]):.4f}')

    if args.record:
        log.save_result(metrics)

    # plotter.record_metric(metrics)
    log('*' * 20 + 'Experiment Success' + '*' * 20)

    return metrics


if __name__ == '__main__':

    args = get_config()
    set_settings(args)
    if args.dataset == 'gpu':
        args.train_device = 'desktop-gpu-gtx-1080ti-fp32'
        args.device_name = '1080Ti'

    # logger plotter
    exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model}, Train_size : {args.train_size}, Bs : {args.bs}, Rank : {args.rank}, "
    exper_detail += f"Train Device : {args.train_device} "
    log_filename = f'Model_{args.model}_{args.dataset}_S{args.train_size}_R{args.rank}_O{args.order}'
    print(log_filename)
    log = Logger(log_filename, exper_detail, args)
    plotter = MetricsPlotter(log_filename, args)
    args.log = log

    # Run Experiment
    RunExperiments(log, args)
    time.sleep(1)
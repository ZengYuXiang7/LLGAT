# coding : utf-8
# Author : yuxiang Zeng
import collections
import time

import numpy as np
import torch
import pickle

from torch.utils.data import DataLoader

from modules.latency_data import get_matrix_and_ops, get_adjacency_and_features

import dgl
from scipy.sparse import csr_matrix

from tqdm import *

from data import experiment, DataModule
from modules.inductive_gnn import OurModel
from utils.config import get_config
from utils.logger import Logger
from utils.metrics import ErrorMetrics
from utils.monitor import EarlyStopping
from utils.plotter import MetricsPlotter
from utils.trainer import get_loss_function, get_optimizer
from utils.utils import optimizer_zero_grad, optimizer_step, lr_scheduler_step, set_settings, set_seed
from utils.utils import makedir
global log, args

torch.set_default_dtype(torch.float32)

import pysnooper


class experiment:
    def __init__(self, args):
        self.args = args

    def load_data(self, args):
        import os
        file_names = os.listdir(args.path + args.dataset)
        pickle_files = [file for file in file_names if file.endswith('.pickle')]
        data = []
        for i in range(len(pickle_files)):
            pickle_file = os.path.join(args.path, args.dataset, pickle_files[i])
            if pickle_files[i].split('.')[-2] in args.train_device:
                with open(pickle_file, 'rb') as f:
                    now = pickle.load(f)
                data.append(now)
                print(pickle_files[i])
        return data

    def preprocess_data(self, data, args):
        x = []
        y = []
        for i in range(len(data)):
            for key, value in data[i].items():
                x.append(list(key))
                y.append(value)
        x = np.array(x)
        y = np.array(y)
        return x, y

    def get_pytorch_index(self, data):
        return torch.as_tensor(data)


# 数据集定义
class DataModule:
    def __init__(self, exper_type, args):
        self.args = args
        self.path = args.path
        self.raw_data = exper_type.load_data(args)
        self.x, self.y = exper_type.preprocess_data(self.raw_data, args)
        self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, self.max_value = self.get_train_valid_test_dataset(self.x, self.y, args)
        self.train_set, self.valid_set, self.test_set = self.get_dataset(self.train_x, self.train_y, self.valid_x, self.valid_y, self.test_x, self.test_y, args)
        self.train_loader, self.valid_loader, self.test_loader = get_dataloaders(self.train_set, self.valid_set, self.test_set, args)
        args.log.only_print(f'Train_length : {len(self.train_loader.dataset)} Valid_length : {len(self.valid_loader.dataset)} Test_length : {len(self.test_loader.dataset)} Max_value : {self.max_value:.2f}')

    def get_dataset(self, train_x, train_y, valid_x, valid_y, test_x, test_y, args):
        return (
            TensorDataset(train_x, train_y, args),
            TensorDataset(valid_x, valid_y, args),
            TensorDataset(test_x, test_y, args)
        )

    def get_train_valid_test_dataset(self, x, y, args):
        x, y = np.array(x),  np.array(y)
        indices = np.random.permutation(len(x))
        x, y = x[indices], y[indices]

        max_value = y.max()
        y /= max_value

        # train_size = int(len(x) * args.density)
        if not args.inductive:
            train_size = int(args.train_size)
            valid_size = int(len(x) * 0.10)
        else:
            train_size = 0
            valid_size = 0

        train_x = x[:train_size]
        train_y = y[:train_size]

        valid_x = x[train_size:train_size + valid_size]
        valid_y = y[train_size:train_size + valid_size]

        test_x = x[train_size + valid_size:]
        test_y = y[train_size + valid_size:]

        return train_x, train_y, valid_x, valid_y, test_x, test_y, max_value


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, args):
        self.args = args
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        key, value = self.x[idx], self.y[idx]
        value = torch.tensor(value).float()
        graph, label = get_matrix_and_ops(key)
        graph, features = get_adjacency_and_features(graph, label)
        graph = dgl.from_scipy(csr_matrix(graph))
        graph = dgl.to_bidirected(graph)
        features = torch.tensor(features).long()
        features = torch.argmax(features, dim=1)
        return graph, features, key, value

class NewDeviceData:
    def __init__(self, exper, args):
        self.args = args
        raw_data = exper.load_data(args)
        self.x, self.y = exper.preprocess_data(raw_data, args)
        print(self.x.shape, self.y.shape)
        self.max_value = np.max(self.y)
        self.y /= self.max_value
        self.test_set = TensorDataset(self.x, self.y, args)
        # self.max_workers = 3
        self.test_loader = DataLoader(
            self.test_set,
            batch_size=256,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=lambda batch: custom_collate_fn(batch, args),
            # num_workers=self.max_workers,
        )


def custom_collate_fn(batch, args):
    from torch.utils.data.dataloader import default_collate
    graph, features, key, values = zip(*batch)
    graph = dgl.batch(graph)
    features = default_collate(features)
    key = default_collate(key)
    values = default_collate(values)
    return graph, features, key, values


def get_dataloaders(train_set, valid_set, test_set, args):
    # max_workers = multiprocessing.cpu_count()
    max_workers = 3
    # torch.set_num_threads()
    train_loader = DataLoader(
        train_set,
        batch_size=args.bs,
        drop_last=False,
        shuffle=True,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        num_workers=max_workers,
        # prefetch_factor=4
    )
    valid_loader = DataLoader(
        valid_set,
        batch_size=args.bs,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        num_workers=max_workers,
        # prefetch_factor=4
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.bs,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        num_workers=max_workers,
        # prefetch_factor=4
    )
    return train_loader, valid_loader, test_loader

# @pysnooper.snoop()
class Model(torch.nn.Module):
    def __init__(self, data, args):
        super().__init__()
        self.args = args
        self.data = data
        self.input_size = data.x.shape[-1]
        self.hidden_size = args.rank
        self.model = OurModel(data, args)

    def forward(self, adjacency, features, key):
        y = self.model.forward(adjacency, features, key)
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

        for train_Batch in dataModule.train_loader:
            try:
                graph, features, key, value = train_Batch
                graph, features, key, value = graph.to(self.args.device), features.to(self.args.device), key.to(self.args.device), value.to(self.args.device)

                # 打印输入数据以检查其有效性
                print(f'Graph: {graph}')
                print(f'Features: {features}')
                print(f'Key: {key}')
                print(f'Value: {value}')

                pred = self.forward(graph, features, key)
                loss = self.loss_function(pred, value)

                # 检查CUDA内存
                print(torch.cuda.memory_summary(device=None, abbreviated=False))

                # 检查梯度爆炸
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f'Gradient of {name}: {param.grad.data.norm()}')

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            except Exception as e:
                print(f"An error occurred during training batch: {e}")
                continue

        t2 = time.time()
        self.eval()
        torch.set_grad_enabled(False)
        time.sleep(0.1)
        return loss, t2 - t1

    def evaluate_one_epoch(self, dataModule, mode='valid'):
        val_loss = 0.
        preds = []
        reals = []
        dataloader = dataModule.valid_loader if mode == 'valid' else dataModule.test_loader
        for batch in (dataloader):
            graph, features, key, value = batch
            graph, features, key, value = graph.to(self.args.device), features.to(self.args.device), key.to(self.args.device), value.to(self.args.device)
            pred = self.forward(graph, features, key)
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


def train_model(runId, log):
    # Set seed
    set_seed(args.seed + runId)

    # Initialize
    exper = experiment(args)
    datamodule = DataModule(exper, args)
    model = Model(datamodule, args)
    monitor = EarlyStopping(args)
    makedir(f'./checkpoints/{args.model}')
    model_path = f'./checkpoints/{args.model}/{log_filename}_round_{runId}.pt'

    try:
        model.setup_optimizer(args)
        model.load_state_dict(torch.load(model_path))
        results = model.evaluate_one_epoch(datamodule, 'test')
        log.only_print(f'MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f}')
        log.only_print(f"Acc = [1%={results['Acc_1']:.4f}, 5%={results['Acc_5']:.4f}, 10%={results['Acc_10']:.4f}] ")
        args.record = False
    except:
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
    return model

def add_new_device(graph, datamodule, idx_map):
    new_device_num = 1
    new_device_idx = graph.num_nodes()
    graph = dgl.add_nodes(graph, new_device_num)
    for i in range(len(datamodule.train_x)):
        dnn_idx = idx_map[tuple(datamodule.train_x[i])]
        if not graph.has_edges_between(dnn_idx, new_device_idx):
            graph.add_edges(dnn_idx, new_device_idx)
    return graph

# 进行 inductive 过程
def eval_inductive(model):
    args.train_device = 'desktop-cpu-core-i7-12700h-fp32'
    args.device_name = 'i7-12700h'
    exper = experiment(args)
    # 首先添加新的计算节点
    model.model.dnn_device_graph = add_new_device(model.model.dnn_device_graph, model.data, model.model.idx_map)
    print('Number of nodes :', model.model.dnn_device_graph.num_nodes())
    print('Number of edges :', model.model.dnn_device_graph.num_edges())

    # 并且先简单获得原始节点,先获得 llm 信息，先编码出原始信息
    with open(f'./agu/{args.device_name}.pkl', 'rb') as f:
        aug_data = pickle.load(f)
        norm = [7, 32, 32, 40, 100]
        for i in range(len(norm)):
            aug_data[i] /= norm[i]
        model.model.aug_data = torch.tensor(aug_data).float().unsqueeze(0)
    print(f'新计算节点 {args.device_name} LLM信息: {model.model.aug_data}')
    new_device_embeds = model.model.info_encoder(model.model.aug_data.to(args.device))
    model.model.dnn_device_graph.ndata['feats'][-1] = new_device_embeds

    # 然后聚合出新计算节点的 embedding
    graph, feats = model.model.dnn_device_graph, model.model.dnn_device_graph.ndata['feats']
    agg_new_embedding = model.model.graph_sage(graph, feats)
    agg_new_embedding = model.model.norm(agg_new_embedding)
    agg_new_embedding = model.model.act(agg_new_embedding)

    
    # 再准备好新的数据集，并测试
    print(graph)
    datamodule = NewDeviceData(exper, args)
    model.setup_optimizer(args)
    reals, preds = [], []
    for batch in tqdm(datamodule.test_loader):
        graph, features, key, value = batch
        graph, features = graph.to(args.device), features.to(args.device)
        new_device_embeds = agg_new_embedding[-1].expand(features.shape[0], -1).to(args.device)
        # 采用agg embeddings测试
        pred = model.model.inference(graph, features, new_device_embeds)
        preds.append(pred)
        reals.append(value)
    reals = torch.cat(reals, dim=0)
    preds = torch.cat(preds, dim=0)
    results = ErrorMetrics(reals * datamodule.max_value, preds * datamodule.max_value, args)
    log.only_print(f'MAE={results["MAE"]:.4f} RMSE={results["RMSE"]:.4f} NMAE={results["NMAE"]:.4f} NRMSE={results["NRMSE"]:.4f}')
    log.only_print(f"Acc = [1%={results['Acc_1']:.4f}, 5%={results['Acc_5']:.4f}, 10%={results['Acc_10']:.4f}] ")
    return results

def RunOnce(args, runId, log):
    args.train_size = 3600
    model = train_model(runId, log)
    results = eval_inductive(model)
    return results


def RunExperiments(log, args):
    log('*' * 20 + 'Experiment Start' + '*' * 20)
    metrics = collections.defaultdict(list)
    results = RunOnce(args, 99, log)
    for key in results:
        metrics[key].append(results[key])
    log(f'{key}: {np.mean(metrics[key]):.4f}')
    log('*' * 20 + 'Experiment Success' + '*' * 20)
    return metrics


if __name__ == '__main__':
    args = get_config()
    set_settings(args)
    # logger plotter
    exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model}, Train_size : {args.train_size}, Bs : {args.bs}, Rank : {args.rank}"
    log_filename = f'Model_{args.model}_Trainsize{args.train_size}_Rank{args.rank}'
    log = Logger(log_filename, exper_detail, args)
    plotter = MetricsPlotter(log_filename, args)
    args.log = log
    log(str(args.__dict__))

    # Run Experiment
    RunExperiments(log, args)


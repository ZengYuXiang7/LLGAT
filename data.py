# coding : utf-8
# Author : yuxiang Zeng
import platform
import numpy as np
import torch
import pickle

from torch.utils.data import DataLoader

from baselines.dnnperf import create_sample_graph
from modules.latency_data import get_matrix_and_ops, get_adjacency_and_features, get_arch_str_from_arch_vector
from utils.config import get_config
from utils.logger import Logger
from utils.plotter import MetricsPlotter
from utils.utils import set_settings

import dgl
from scipy.sparse import csr_matrix


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
        if self.args.model == 'ours':
            # print('-'*100)
            # print(key)
            # print(get_arch_str_from_arch_vector(key))
            graph, label = get_matrix_and_ops(key)
            graph, features = get_adjacency_and_features(graph, label)
            graph = dgl.from_scipy(csr_matrix(graph))
            graph = dgl.to_bidirected(graph)
            features = torch.tensor(features).long()
            key = torch.argmax(features, dim=1)
            # print(graph)
            # print(graph.edges())
            # print(key)
            # exit()
            return graph, key, value
        elif self.args.model == 'gcn':
            graph, label = get_matrix_and_ops(key)
            graph, features = get_adjacency_and_features(graph, label)
            graph = dgl.from_scipy(csr_matrix(graph))
            graph = dgl.to_bidirected(graph)
            features = torch.tensor(features).long()
            features = torch.argmax(features, dim=1)
            return graph, features, value
        elif self.args.model in ['brp_nas', 'help']:
            graph, label = get_matrix_and_ops(key)
            graph, features = get_adjacency_and_features(graph, label)
            graph = torch.tensor(graph).float()
            features = torch.tensor(features).float()
            return graph, features, value
        elif self.args.model in ['mlp', 'lstm', 'gru', 'flops']:
            key = torch.tensor(key).long()
            return None, key, value
        elif self.args.model == 'birnn':
            key = torch.tensor(key).float()
            return None, key, value
        elif self.args.model == 'dnnperf':
            graph, label = get_matrix_and_ops(key)
            graph, features = get_adjacency_and_features(graph, label)
            graph = graph[1:, 1:]
            graph = dgl.from_scipy(csr_matrix(graph))
            key = torch.argmax(torch.tensor(features).long(), dim=1)[1:]
            graph = create_sample_graph(key, graph, self.args)
            return graph, key, value


def custom_collate_fn(batch, args):
    from torch.utils.data.dataloader import default_collate
    graphs, features, values = zip(*batch)
    if args.model == 'ours':
        # batched_graph = default_collate(graphs)
        batched_graph = dgl.batch(graphs)
    elif args.model == 'gcn':
        batched_graph = dgl.batch(graphs)
    elif args.model == 'dnnperf':
        batched_graph = dgl.batch(graphs)
    elif args.model in ['brp_nas', 'help']:
        batched_graph = default_collate(graphs)
    elif args.model in ['mlp', 'lstm', 'gru', 'birnn']:
        batched_graph = torch.zeros(len(features))
    features = default_collate(features)
    values = default_collate(values)
    return batched_graph, features, values


def get_dataloaders(train_set, valid_set, test_set, args):
    # 检查操作系统
    if platform.system() == "Darwin":  # "Darwin" 是 macOS 的系统标识
        max_workers = 0
    else:
        max_workers = 6
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
        batch_size=args.bs * 8,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        num_workers=max_workers,
        # prefetch_factor=4
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.bs * 8,
        drop_last=False,
        shuffle=False,
        pin_memory=True,
        collate_fn=lambda batch: custom_collate_fn(batch, args),
        num_workers=max_workers,
        # prefetch_factor=4
    )
    return train_loader, valid_loader, test_loader



if __name__ == '__main__':
    args = get_config()
    set_settings(args)

    # logger plotter
    exper_detail = f"Dataset : {args.dataset.upper()}, Model : {args.model}, Train_size : {args.train_size}"
    log_filename = f'{args.train_size}_r{args.dimension}'
    log = Logger(log_filename, exper_detail, args)
    plotter = MetricsPlotter(log_filename, args)
    args.log = log
    log(str(args.__dict__))

    exper = experiment(args)
    datamodule = DataModule(exper, args)
    for train_batch in datamodule.train_loader:
        graph, features, y = train_batch
        print(graph.shape, features.shape, y.shape)
        break
    print('Done!')

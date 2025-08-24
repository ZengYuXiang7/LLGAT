# coding : utf-8
# Author : Anonymous
import numpy as np
import torch
import dgl
import pickle

from modules.chatgpt import NAS_ChatGPT
from modules.pred_layer import Predictor

class GraphReader(torch.nn.Module):
    def __init__(self, input_dim, rank, order, args):
        super(GraphReader, self).__init__()
        self.args = args
        self.rank = rank
        self.order = order
        self.dnn_embedding = torch.nn.Embedding(6, rank)
        self.layers = torch.nn.ModuleList([dgl.nn.pytorch.SAGEConv(rank, rank, aggregator_type='gcn') for i in range(order)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(rank) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(order)])
        self.dropout = torch.nn.Dropout(0.10)

    def forward(self, graph, features):
        g, feats = graph, self.dnn_embedding(features).reshape(features.shape[0] * 9, -1)
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats)
            feats = norm(feats)
            feats = act(feats)
            feats = self.dropout(feats)
        batch_sizes = torch.as_tensor(g.batch_num_nodes()).to(self.args.device)  # 每个图的节点数
        first_nodes_idx = torch.cumsum(torch.cat((torch.tensor([0]).to(self.args.device), batch_sizes[:-1])), dim=0)  # 使用torch.cat来连接Tensor
        first_node_features = feats[first_nodes_idx]
        return first_node_features

class DnnDeviceGNN(torch.nn.Module):
    def __init__(self, args):
        super(DnnDeviceGNN, self).__init__()
        self.args = args
    def forward(self, graph, features):
        pass

def get_dnn_to_idx():
    all_nodes = []
    idx_map = {}

    def dfs(arr, step):
        if step >= 6:
            combination = tuple(arr)  # Use tuple since lists are not hashable and cannot be dictionary keys
            if combination not in idx_map:
                idx = len(idx_map)
                idx_map[combination] = idx
                all_nodes.append(combination)
            return

        for i in range(5):
            arr[step] = i
            dfs(arr, step + 1)
            arr[step] = 0

    dfs([0, 0, 0, 0, 0, 0], 0)

    return idx_map


def add_new_device(graph, datamodule, idx_map):
    new_device_num = 1
    new_device_idx = graph.num_nodes()
    graph = dgl.add_nodes(graph, new_device_num)
    for i in range(len(datamodule.train_x)):
        dnn_idx = idx_map[tuple(datamodule.train_x[i])]
        if not graph.has_edges_between(dnn_idx, new_device_idx):
            graph.add_edges(dnn_idx, new_device_idx)
    return graph


class OurModel(torch.nn.Module):
    def __init__(self, datamodule, args):
        super(OurModel, self).__init__()
        self.args = args
        self.rank = args.rank
        self.graph_encoder = GraphReader(6, args.rank, args.order, self.args)
        # 添加 算力任务--计算节点图
        # 首先获得算力任务到 idx 的映射表
        self.idx_map = get_dnn_to_idx()
        # 然后构建图
        self.dnn_device_graph = dgl.graph([])
        self.dnn_device_graph = dgl.add_nodes(self.dnn_device_graph, len(self.idx_map))
        # 添加首个计算节点
        self.dnn_device_graph = add_new_device(self.dnn_device_graph, datamodule, self.idx_map).to(self.args.device)
        print(self.dnn_device_graph)

        self.graph_sage = dgl.nn.pytorch.SAGEConv(args.rank, args.rank, aggregator_type='gcn')
        self.norm = torch.nn.LayerNorm(args.rank)
        self.act = torch.nn.ReLU()

        # 大预言模型部分
        self.info_encoder = torch.nn.Linear(5, args.rank)
        try:
            with open(f'./agu/{args.device_name}.pkl', 'rb') as f:
                self.aug_data = torch.tensor(pickle.load(f)).float().unsqueeze(0)
        except:
            import ast
            llm = NAS_ChatGPT(args)
            self.aug_data = ast.literal_eval(llm.get_device_more_info(args.device_name))
            self.aug_data = torch.tensor(self.aug_data).float().unsqueeze(0)
            print("增强数据: ", self.aug_data)
        self.predictor = Predictor(self.rank * 2, self.rank, 1)

    def copy_dnn_features_to_new_graph(self, graph_embeds, key):
        for i in range(len(key)):
            idx = self.idx_map[tuple(np.array(key[i].cpu()))]
            self.dnn_device_graph.ndata['feats'][idx] = graph_embeds[i]

    def copy_device_features_to_new_graph(self, info_embeds):
        self.dnn_device_graph.ndata['feats'][-1] = info_embeds

    def forward(self, graph, features, key, ):
        # 第零步：初始化
        self.dnn_device_graph.ndata['feats'] = torch.zeros((self.dnn_device_graph.num_nodes(), self.args.rank)).to(self.args.device)
        # 第一步：先复制到DNN特征到新图上
        graph_embeds = self.graph_encoder(graph, features)
        self.copy_dnn_features_to_new_graph(graph_embeds, key)
        # 第二步：再复制设备特征到新图上
        info_embeds = self.info_encoder(self.aug_data.to(self.args.device))
        self.copy_device_features_to_new_graph(info_embeds)
        # 第三步：执行聚合器训练
        g, feats = self.dnn_device_graph, self.dnn_device_graph.ndata['feats']
        feats = self.graph_sage(g, feats)
        feats = self.norm(feats)
        feats = self.act(feats)

        info_embeds = feats[-1].expand(features.shape[0], -1).to(self.args.device)
        # key -> whole_key -> features
        embeds = torch.cat([graph_embeds, info_embeds], dim=1)
        y = self.predictor(embeds)
        return y

    def inference(self, graph, features, info_embeds):
        graph_embeds = self.graph_encoder(graph, features)
        embeds = torch.cat([graph_embeds, info_embeds], dim=1)
        y = self.predictor(embeds)
        return y

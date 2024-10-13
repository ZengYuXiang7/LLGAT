# coding : utf-8
# Author : yuxiang Zeng

import torch
import dgl
from dgl.nn.pytorch import SAGEConv
import pickle

from modules.chatgpt import NAS_ChatGPT
from modules.pred_layer import Predictor
from utils.config import get_config


class GraphEncoder(torch.nn.Module):
    def __init__(self, input_dim, rank, order, args):
        super(GraphEncoder, self).__init__()
        self.args = args
        self.rank = rank
        self.order = order
        """
            0 con1 1 con3 2 max3 3 input 4 output 5 None
            0 con1 1 con3 2 max3 3 input 4 output 5 global node
        """
        if args.op_encoder == 'embed':
            self.dnn_embedding = torch.nn.Embedding(6, rank)
        elif args.op_encoder == 'value':
            self.transfer = torch.nn.Linear(1, rank)
        elif args.op_encoder == 'one_hot':
            self.transfer = torch.nn.Linear(6, rank)

        # Graph Encoder
        if args.graph_encoder == 'gcn':
            self.layers = torch.nn.ModuleList([dgl.nn.pytorch.GraphConv(rank, rank) for i in range(order)])
        elif args.graph_encoder == 'graphsage':
            self.layers = torch.nn.ModuleList([dgl.nn.pytorch.SAGEConv(rank, rank, aggregator_type='gcn') for i in range(order)])
        elif args.graph_encoder == 'gat':
            self.layers = torch.nn.ModuleList([dgl.nn.pytorch.GATConv(rank, rank, args.heads, 0.10) for i in range(order)])
        else:
            raise NotImplementedError
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(rank) for _ in range(order)])
        self.acts = torch.nn.ModuleList([torch.nn.ReLU() for _ in range(order)])
        self.dropout = torch.nn.Dropout(0.10)

    def forward(self, graph, features):
        if self.args.op_encoder == 'embed':
            feats = self.dnn_embedding(features).reshape(features.shape[0] * 9, -1)
        elif self.args.op_encoder == 'one_hot':
            bs = features.shape[0]
            feats = torch.nn.functional.one_hot(features, num_classes=6).float().reshape(bs * 9, -1)
            feats = self.transfer(feats)
        elif self.args.op_encoder == 'value':
            bs = features.shape[0]
            feats = features / 5
            feats = feats.reshape(bs * 9, 1)
            feats = self.transfer(feats)
        else:
            raise NotImplementedError
        g = graph
        for i, (layer, norm, act) in enumerate(zip(self.layers, self.norms, self.acts)):
            feats = layer(g, feats)
            if self.args.graph_encoder == 'gat':
                feats = feats.mean(dim=1)  # 聚合多个头的输出
            feats = norm(feats)
            feats = act(feats)
            if self.args.graph_encoder != 'gat':
                feats = self.dropout(feats)
        batch_sizes = torch.as_tensor(g.batch_num_nodes()).to(self.args.device)  # 每个图的节点数
        first_nodes_idx = torch.cumsum(torch.cat((torch.tensor([0]).to(self.args.device), batch_sizes[:-1])), dim=0)  # 使用torch.cat来连接Tensor
        first_node_features = feats[first_nodes_idx]
        return first_node_features


class LLMs(torch.nn.Module):
    def __init__(self, args):
        super(LLMs, self).__init__()
        self.args = args
        try:
            with open(f'./agu/{args.device_name}.pkl', 'rb') as f:
                self.aug_data = pickle.load(f)
                norm = [7, 32, 32, 40, 100] if args.dataset == 'cpu' else [9000, 2000, 16, 400]
                for i in range(len(norm)):
                    self.aug_data[i] /= norm[i]
                self.aug_data = torch.tensor(self.aug_data).float().unsqueeze(0)
            print('加载存储数据', self.aug_data)
        except:
            import ast
            llm = NAS_ChatGPT(args)
            self.aug_data = ast.literal_eval(llm.get_device_more_info(args.device_name, args.dataset))
            with open(f'./agu/{args.device_name}_new.pkl', 'wb') as f:
                pickle.dump(self.aug_data, f)
            norm = [7, 32, 32, 40, 100] if args.dataset == 'cpu' else [9000, 2000, 16, 400]
            for i in range(len(norm)):
                self.aug_data[i] /= norm[i]
            self.aug_data = torch.tensor(self.aug_data).float().unsqueeze(0)
            print("LLM: ", self.aug_data)
        # Encoder 
        self.info_encoder = torch.nn.Linear(5 if args.dataset == 'cpu' else 4, 100)

    def forward(self, batch_size):
        aug_data_repeated = self.aug_data.expand(batch_size, -1).to(self.args.device)
        info_embeds = self.info_encoder(aug_data_repeated)
        return info_embeds
    

class OurModel(torch.nn.Module):
    def __init__(self, args):
        super(OurModel, self).__init__()
        self.args = args
        self.rank = args.rank
        self.graph_encoder = GraphEncoder(9, args.rank, args.order, self.args)
        if args.llm:
            self.llms = LLMs(args)
            self.predictor = Predictor(self.rank + 100, self.rank, 1)
            if args.att:
                if args.cross:
                    print('Cross')
                    self.cross_attention = CrossAttention(args.rank, args.rank, num_heads=args.heads)
                    self.predictor = Predictor(self.rank * 2, self.rank, 1)
                else:
                    print('Self')
                    self.self_attention = torch.nn.MultiheadAttention(embed_dim=args.rank, num_heads=args.heads, dropout=0.00)
                    self.predictor = Predictor(self.rank * 2, self.rank, 1)
        else:
            self.pred_layers = torch.nn.Linear(args.rank, 1)

    def forward(self, graph, features):
        if self.args.llm:
            y = self.graph_with_llm(graph, features)
        else:
            y = self.only_graph(graph, features)
        return y

    def graph_with_llm(self, graph, features):
        info_embeds = self.llms(features.shape[0])
        graph_embeds = self.graph_encoder(graph, features)
        # print(info_embeds.shape, graph_embeds.shape)

        # Only Concatenate
        if not self.args.att:
            embeds = torch.cat([graph_embeds, info_embeds], dim=1)
            y = self.predictor(embeds)
        else:
            # print(self.args.cross, self.args.att)
            if self.args.cross == 1:
                # Cross Attention
                graph_embeds = graph_embeds.unsqueeze(0)
                info_embeds = info_embeds.unsqueeze(0)
                embeds = self.cross_attention(graph_embeds, info_embeds)
                # y = self.cross_attention(graph_embeds, info_embeds)
                y = self.predictor(embeds)
                # print('cross')
            else:
                # Self Attention
                embeds = torch.stack([graph_embeds, info_embeds], dim=0)
                embeds, _ = self.self_attention(embeds, embeds, embeds)
                # embeds = torch.mean(embeds, dim=0)
                embeds = embeds.permute(1, 0, 2)
                embeds = embeds.reshape(len(embeds), -1)
                y = self.predictor(embeds)

                # print('self')
        return y

    def only_graph(self, graph, features):
        graph_embeds = self.graph_encoder(graph, features)
        y = self.pred_layers(graph_embeds)
        return y


if __name__ == '__main__':
    # Build a random graph
    args = get_config()
    num_nodes, num_edges = 100, 200
    src_nodes = torch.randint(0, num_nodes, (num_edges,))
    dst_nodes = torch.randint(0, num_nodes, (num_edges,))
    print(src_nodes.shape, dst_nodes.shape)

    graph = dgl.graph((src_nodes, dst_nodes))
    graph = dgl.add_self_loop(graph)

    # Demo test
    bs = 32
    features = torch.randn(num_nodes, 64)
    graph_gcn = GraphSAGEConv(64, 128, 2, args)
    # print(graph_gcn)
    embeds = graph_gcn(graph, features)
    print(embeds.shape)

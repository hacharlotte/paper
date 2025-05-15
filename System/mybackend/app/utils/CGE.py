from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import copy
import math
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import Linear
from torch_geometric.nn.inits import glorot, reset
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.nn import GATConv

def group(xs: List[Tensor], beta_1, beta_2) -> Optional[Tensor]:
    if len(xs) == 0:
        return None
    else:
        out = torch.stack(xs)
        if out.numel() == 0:
            return out.view(0, out.size(-1))
        n,d = out.shape[1:]
        final_out = torch.zeros(n, d).cuda()
        mask = torch.all(out[1] == 0, dim=1).cuda()
        final_out[mask] = out[0, mask]
        final_out[~mask] = out[0, ~mask] * beta_1 + out[1, ~mask] * beta_2
        return final_out


class GAT(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads, num_layers):
        super(GAT, self).__init__()

        # 定义GAT层
        self.gat_layers = nn.ModuleList()

        # 第一层GAT
        self.gat_layers.append(GATConv(in_channels, out_channels, heads=num_heads, dropout=0.6))

        # 后续GAT层（如果有的话）
        for _ in range(num_layers - 1):
            self.gat_layers.append(GATConv(out_channels * num_heads, out_channels, heads=num_heads, dropout=0.6))

        # 输出层
        self.fc = nn.Linear(out_channels * num_heads, out_channels)

    def forward(self, x, edge_index):
        """
        前向传播
        :param x: 输入节点特征 (num_nodes, in_channels)
        :param edge_index: 图的邻接矩阵（边列表） (2, num_edges)
        :return: 输出节点特征 (num_nodes, out_channels)
        """
        for gat_layer in self.gat_layers:
            x = F.elu(gat_layer(x, edge_index))

        # 通过全连接层转换为最终的输出
        x = self.fc(x)
        return x
class CGEConv(MessagePassing):
    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        negative_slope=0.2,
        dropout: float = 0.0,
        beta_intra=0.7,
        beta_inter=0.3,
        **kwargs,
    ):
        super().__init__(aggr='add', node_dim=0, **kwargs)

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.heads = heads
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.metadata = metadata
        self.dropout = dropout
        self.beta_intra = beta_intra
        self.beta_inter = beta_inter

        self.proj = nn.ModuleDict()
        for node_type, in_channels in self.in_channels.items():
            self.proj[node_type] = Linear(in_channels, out_channels)

        dim = out_channels // heads
        self.lin_src = nn.Parameter(torch.Tensor(1, heads, dim))
        self.lin_dst = nn.Parameter(torch.Tensor(1, heads, dim))

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.proj)
        glorot(self.lin_src)
        glorot(self.lin_dst)

    def forward(
        self, x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj]) -> Dict[NodeType, Optional[Tensor]]:
        """
        Args:
            x_dict (Dict[str, Tensor]): A dictionary holding input node
                features for each individual node type.
            edge_index_dict (Dict[str, Union[Tensor, SparseTensor]]): A
                dictionary holding graph connectivity information for each
                individual edge type.

        :rtype: :obj:`Dict[str, Optional[Tensor]]` - The output node embeddings
            for each node type.
        """
        # Combine 'intra' and 'inter' edges into a single list for the same edge type
        if ('event', 'inter', 'event') in edge_index_dict:
            combined_edge_index = torch.cat([
                edge_index_dict[('event', 'intra', 'event')],
                edge_index_dict[('event', 'inter', 'event')]
            ], dim=1)
        else:
            combined_edge_index = edge_index_dict[('event', 'intra', 'event')]

        H, D = self.heads, self.out_channels // self.heads
        x_node_dict, out_dict = {}, {}

        # Iterate over node types:
        for node_type, x in x_dict.items():
            x_node_dict[node_type] = self.proj[node_type](x).view(-1, H, D)
            out_dict[node_type] = []

        # Iterate over combined edge type:
        edge_type = 'event'  # Simplified for same edge type
        lin_src = self.lin_src
        lin_dst = self.lin_dst

        for node_type, x_src in x_node_dict.items():
            alpha_src = (x_src * lin_src).sum(dim=-1)
            alpha_dst = (x_src * lin_dst).sum(dim=-1)
            out = self.propagate(combined_edge_index, x=(x_src, x_src), alpha=(alpha_src, alpha_dst), size=None)
            out = F.relu(out)
            out_dict[node_type].append(out)

        # Iterate over node types:
        for node_type, outs in out_dict.items():
            if len(outs) == 1:
                out_dict[node_type] = outs[0]
            elif len(outs) == 0:
                out_dict[node_type] = None
            else:
                out = group(outs, self.beta_intra, self.beta_inter)
                out_dict[node_type] = out

        return out_dict

    def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
                index: Tensor, ptr: Optional[Tensor],
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.out_channels}, heads={self.heads})'


class GCN(nn.Module):
    def __init__(self,in_dim, mem_dim, num_layers, num_head, args):
        super(GCN, self).__init__()
        self.layers = num_layers
        self.mem_dim = mem_dim
        self.in_dim = in_dim
        self.isatt = args.isatt
        # drop out
        self.gcn_drop = nn.Dropout(0.1)

        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))
        if args.isatt == 1:
                    self.attention_heads = num_head
                    self.head_dim = self.mem_dim // self.layers
                    self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim)
        else:
                    self.mlp = nn.Sequential(nn.Linear((1 * args.n_last + 2 * 768), args.mlp_size),
                                  nn.ReLU(), nn.Dropout(args.mlp_drop),
                                  nn.Linear(args.mlp_size, args.no_of_classes))
        self.weight_list = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.in_dim if layer == 0 else self.mem_dim
            self.weight_list.append(nn.Linear(input_dim, self.mem_dim))


    def forward(self, gcn_inputs, pad_event1_pos,pad_event2_pos, rel_type, event_pairs,node_event):
        if self.isatt == 1:
                    attn_tensor = self.attn(gcn_inputs, gcn_inputs)
                    attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
                    adj_ag = None
        # * Average Multi-head Attention matrixes
                    for i in range(self.attention_heads):
                        if adj_ag is None:
                            adj_ag = attn_adj_list[i]
                        else:
                            adj_ag = adj_ag + attn_adj_list[i]
                    adj_ag /= self.attention_heads

                    for j in range(adj_ag.size(0)):
                        adj_ag[j] -= torch.diag(torch.diag(adj_ag[j]))
                        adj_ag[j] += torch.eye(adj_ag[j].size(0)).cuda()
        else:
            g = gcn_inputs.squeeze(0)
            event1 = torch.index_select(g, 0, pad_event1_pos[0])
            event2 = torch.index_select(g, 0, pad_event2_pos[0])
            event_pair_embed = torch.cat([event1, event2], dim=-1)
            event_diff = event2 - event1
            event_pair_pre = torch.cat((event_pair_embed, event_diff), dim=1)
            prediction = self.mlp(event_pair_pre)
            adj_ag = self.get_graphedge_index(prediction,rel_type, event_pairs,node_event)            
        denom_ag = adj_ag.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs

        for l in range(self.layers):
            Ax = adj_ag.bmm(outputs)
            AxW = self.weight_list[l](Ax)
            AxW = AxW / denom_ag
            gAxW = F.relu(AxW)
            outputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return outputs,adj_ag
    def get_graphedge_index(self,prediction,rel_type, event_pair,node_event):
        predt = torch.softmax(prediction, dim=1)
        causal_matrix = torch.zeros((len(node_event), len(node_event))).cuda()
        for i, (event1, event2) in enumerate(event_pair):
            causal_matrix[event1, event2] = torch.argmax(predt[i]).item()
        # 将矩阵变为对称矩阵
        causal_matrix = causal_matrix + causal_matrix.T

        # 设置对角线为 1
        causal_matrix.fill_diagonal_(1)
        return causal_matrix.unsqueeze(0)



def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        # mask = mask[:, :, :query.size(1)]
        # if mask is not None:
        #     mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn

# class CGEConv(MessagePassing):
#     r"""
#     Args:
#         in_channels (int or Dict[str, int]): Size of each input sample of every
#             node type, or :obj:`-1` to derive the size from the first input(s)
#             to the forward method.
#         out_channels (int): Size of each output sample.
#         metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
#             of the heterogeneous graph, *i.e.* its node and edge types given
#             by a list of strings and a list of string triplets, respectively.
#             See :meth:`torch_geometric.data.HeteroData.metadata` for more
#             information.
#         heads (int, optional): Number of multi-head-attentions.
#             (default: :obj:`1`)
#         negative_slope (float, optional): LeakyReLU angle of the negative
#             slope. (default: :obj:`0.2`)
#         dropout (float, optional): Dropout probability of the normalized
#             attention coefficients which exposes each node to a stochastically
#             sampled neighborhood during training. (default: :obj:`0`)
#         beta_intra: (float, optional): weights of intra edge. (default: :obj:`0.7`)
#         beta_inter: (float, optional): weights of inter edge. (default: :obj:`0.3`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#     def __init__(
#         self,
#         in_channels: Union[int, Dict[str, int]],
#         out_channels: int,
#         metadata: Metadata,
#         heads: int = 1,
#         negative_slope=0.2,
#         dropout: float = 0.0,
#         beta_intra=0.7,
#         beta_inter=0.3,
#         **kwargs,
#     ):
#         super().__init__(aggr='add', node_dim=0, **kwargs)
#
#         if not isinstance(in_channels, dict):
#             in_channels = {node_type: in_channels for node_type in metadata[0]}
#
#         self.heads = heads
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.negative_slope = negative_slope
#         self.metadata = metadata
#         self.dropout = dropout
#         self.beta_intra = beta_intra
#         self.beta_inter = beta_inter
#
#         self.proj = nn.ModuleDict()
#         for node_type, in_channels in self.in_channels.items():
#             self.proj[node_type] = Linear(in_channels, out_channels)
#
#         self.lin_src = nn.ParameterDict()
#         self.lin_dst = nn.ParameterDict()
#         dim = out_channels // heads
#         for edge_type in metadata[1]:
#             edge_type = '__'.join(edge_type)
#             self.lin_src[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
#             self.lin_dst[edge_type] = nn.Parameter(torch.Tensor(1, heads, dim))
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         reset(self.proj)
#         glorot(self.lin_src)
#         glorot(self.lin_dst)
#
#     def forward(
#         self, x_dict: Dict[NodeType, Tensor],
#         edge_index_dict: Dict[EdgeType,
#                               Adj]) -> Dict[NodeType, Optional[Tensor]]:
#         r"""
#         Args:
#             x_dict (Dict[str, Tensor]): A dictionary holding input node
#                 features  for each individual node type.
#             edge_index_dict (Dict[str, Union[Tensor, SparseTensor]]): A
#                 dictionary holding graph connectivity information for each
#                 individual edge type, either as a :obj:`torch.LongTensor` of
#                 shape :obj:`[2, num_edges]` or a
#                 :obj:`torch_sparse.SparseTensor`.
#
#         :rtype: :obj:`Dict[str, Optional[Tensor]]` - The output node embeddings
#             for each node type.
#             In case a node type does not receive any message, its output will
#             be set to :obj:`None`.
#         """
#         H, D = self.heads, self.out_channels // self.heads
#         x_node_dict, out_dict = {}, {}
#
#         # Iterate over node types:
#         for node_type, x in x_dict.items():
#             x_node_dict[node_type] = self.proj[node_type](x).view(-1, H, D)
#             out_dict[node_type] = []
#
#         # Iterate over edge types:
#         for edge_type, edge_index in edge_index_dict.items():
#             src_type, _, dst_type = edge_type
#             edge_type = '__'.join(edge_type)
#             lin_src = self.lin_src[edge_type]
#             lin_dst = self.lin_dst[edge_type]
#             x_src = x_node_dict[src_type]
#             x_dst = x_node_dict[dst_type]
#             alpha_src = (x_src * lin_src).sum(dim=-1)
#
#             alpha_dst = (x_dst * lin_dst).sum(dim=-1)
#             out = self.propagate(edge_index, x=(x_src, x_dst),
#                                  alpha=(alpha_src, alpha_dst), size=None)
#
#             out = F.relu(out)
#             out_dict[dst_type].append(out)
#
#         # iterate over node types:
#         for node_type, outs in out_dict.items():
#             if outs.__len__() == 1:
#                 out_dict[node_type] = outs[0]
#             elif outs.__len__() == 0:
#                 out_dict[node_type] = None
#                 continue
#             else:
#                 out = group(outs, self.beta_intra, self.beta_inter)
#                 out_dict[node_type] = out
#
#         return out_dict
#
#     def message(self, x_j: Tensor, alpha_i: Tensor, alpha_j: Tensor,
#                 index: Tensor, ptr: Optional[Tensor],
#                 size_i: Optional[int]) -> Tensor:
#
#         alpha = alpha_j + alpha_i
#         alpha = F.leaky_relu(alpha, self.negative_slope)
#         alpha = softmax(alpha, index, ptr, size_i)
#         alpha = F.dropout(alpha, p=self.dropout, training=self.training)
#         out = x_j * alpha.view(-1, self.heads, 1)
#         return out.view(-1, self.out_channels)
#
#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}({self.out_channels}, '
#                 f'heads={self.heads})')

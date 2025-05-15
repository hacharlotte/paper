# coding: UTF-8
import numpy as np
import torch
import copy
import math
import torch.nn as nn
from transformers import BertModel
import torch.nn.functional as F
from .CGE import GAT,GCN
from torch.nn.utils.rnn import pad_sequence
from .TransformerModel import TransformerModel
embedding_size = 768


def create_weight_matrix(tensor1_value, tensor2_value, list1, list2, node_count=12):
    # 初始化 node_count x node_count 的全零 tensor
    weight_matrix = torch.zeros((node_count, node_count), dtype=torch.float32)

    # 处理 list1 和 tensor1_value，填充权重矩阵
    for i in range(len(tensor1_value)):
        if i < len(list1):  # 确保 tensor1_value 的索引在 list1 的范围内
            source1, target1 = list1[i]
            if source1 < node_count and target1 < node_count:
                weight_matrix[source1, target1] = tensor1_value[i]

    # 处理 list2 和 tensor2_value，填充权重矩阵
    for i in range(len(tensor2_value)):
        if i < len(list2):  # 确保 tensor2_value 的索引在 list2 的范围内
            source2, target2 = list2[i]
            if source2 < node_count and target2 < node_count:
                weight_matrix[source2, target2] = tensor2_value[i]

    return weight_matrix


def get_matching_indices_for_value(tensor_index,tensor_value, node_pairs, n):
    """
    获取 tensor1 中值为指定值的索引，并从给定的node_pairs中找出对应的 [源节点, 目标节点] 对。

    Parameters:
    - tensor1: 一个大小为300的tensor，包含值为0、1或2
    - node_pairs: 一个大小为300x2的列表，每一行包含 [源节点索引, 目标节点索引]
    - value: 要匹配的值，默认为1

    Returns:
    - matched_pairs: 匹配的 [源节点, 目标节点] 对列表
    """

    # 获取 tensor1 中值为指定值的所有索引
    matching_indices_1 = (tensor_index == 1)
    tensor_max_value = tensor_value[torch.arange(len(tensor_index)), tensor_index]
    tensor1_value = tensor_max_value[matching_indices_1]

    # 获取对应的索引位置
    matching_indices_1 = matching_indices_1.nonzero(as_tuple=True)[0]

    # 通过这些索引获取匹配的 [源节点, 目标节点] 对
    matched_pairs_1 = [node_pairs[idx] for idx in matching_indices_1]

    # 获取 tensor2 中值为指定值的所有索引
    matching_indices_2 = (tensor_index == 2)
    tensor2_value = tensor_max_value[matching_indices_2]

    # 获取对应的索引位置
    matching_indices_2 = matching_indices_2.nonzero(as_tuple=True)[0]

    # 通过这些索引获取匹配的 [源节点, 目标节点] 对
    matched_pairs_2 = [node_pairs[idx] for idx in matching_indices_2]

    weight_matrix = create_weight_matrix(tensor1_value, tensor2_value, list1=matched_pairs_1, list2=matched_pairs_2,
                                         node_count=n)

    return weight_matrix.unsqueeze(0).cuda()
def compute_attention_matrix(inputs):
    """
    计算注意力矩阵，输入是len x dim的矩阵，输出是len x len的注意力矩阵
    :param inputs: 输入矩阵，形状为(len, dim)
    :return: 注意力矩阵，形状为(len, len)
    """
    # 计算点积
    attention_matrix = np.matmul(inputs, inputs.T)

    # 进行归一化操作（可选，根据需要）
    attention_matrix = attention_matrix / np.sqrt(inputs.shape[1])  # 除以sqrt(dim)，以防止梯度消失问题

    return attention_matrix
class focal_loss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2, num_classes=3, size_average=True):
        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += (1 - alpha)
            self.alpha[1:] += alpha
        self.gamma = gamma

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=-1)
        preds_logsoft = F.log_softmax(preds, dim=-1)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),preds_logsoft)
        loss = torch.mul(alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

class bertCSRModel_nosem(nn.Module):
    def __init__(self, args):
        super(bertCSRModel_nosem, self).__init__()
        self.device = args.device
        self.pretrained_model = BertModel.from_pretrained(args.model_name)
        self.mlp = nn.Sequential(nn.Linear((1 * args.n_last + 2 * embedding_size) * 2, args.mlp_size),
                                  nn.ReLU(), nn.Dropout(args.mlp_drop),
                                  nn.Linear(args.mlp_size, args.no_of_classes))
        self.gcn_layers = nn.ModuleList()
        # self.gcn_layers.append(GCN(768, 768, args.GCN_layers, args.GCN_heads, args))
        for i in range(int(args.max_iteration / 2)):
            self.gcn_layers.append(GCN(768, 768, args.GCN_layers, args.GCN_heads,  args))
        self.rate = args.rate
        self.w = args.w
        self.max_iteration = args.max_iteration
        self.threshold = args.threshold
        self.norm = nn.LayerNorm(embedding_size)
        self.min_iteration = args.min_iteration
        if args.iscat == 1:
                    self.transformer_encoder = TransformerModel(768,768,args.Trans_head,args.Trans_layers,3072,512,0.1)##6 12达到最佳 drop 0.2 0.1
        self.liner_sem = nn.Linear(768, 768)
        self.liner_str = nn.Linear(768, 768)
        self.liner_event = nn.Linear(768 * 2, 768)
        self.penal1 = args.penal
        self.focal_loss = focal_loss(gamma=args.gamma, num_classes=args.no_of_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss_type = args.loss_type
        self.iscat = args.iscat
    def forward(self, enc_input_ids, enc_mask_ids, node_event, t1_pos, t2_pos, target, rel_type, event_pairs):
        sent_emb = self.pretrained_model(enc_input_ids, enc_mask_ids)[0].to(self.device)
        sent_list = []
        for idx, event in enumerate(node_event):
            sent_id = int(event[0])
            event_pid = event[1]
            e_emb = self.extract_event(sent_emb[sent_id], event_pid).to(self.device)
            if idx == 0:
                event_embed = e_emb
            else:
                event_embed = torch.cat((event_embed, e_emb))
            if sent_id not in sent_list:
                sent_list.append(sent_id)
        if len(event_pairs) > 1:
            target = torch.cat([torch.tensor(t) for t in target], dim=0).to(self.device)
        else:
            target = torch.tensor(target[0]).to(self.device)
#         if self.max_iteration > sent_list.__len__():
#             max_iteration = sent_list.__len__()
#         else:
#             max_iteration = self.max_iteration
        max_iteration = self.max_iteration
        ##未添加relu
        event_sem_input = self.liner_sem(event_embed)
        event_sem_input = F.relu(event_sem_input)
        event_str_input = self.liner_str(event_embed)
        # event_str_input = self.CGE.proj['event'](event_embed)
        event_str_input = F.relu(event_str_input)
        if self.iscat == 1:
                    event_sem_output, adj_sem = self.transformer_encoder(event_sem_input)
                    adj_sem[0] -= torch.diag(torch.diag(adj_sem[0]))
                    adj_sem[0] += torch.eye(adj_sem[0].size(0)).cuda()
        pad_event1_pos = pad_sequence([torch.tensor(pos) for pos in t1_pos]).t().to(self.device).long()
        pad_event2_pos = pad_sequence([torch.tensor(pos) for pos in t2_pos]).t().to(self.device).long()

        iteration = 0
        difference = self.threshold
        loss = 0.0
        penal = 0.0
        gcn_inputs = event_str_input.unsqueeze(0)
        if self.iscat == 1:
                   gcn_sem_dis = event_sem_output.unsqueeze(0)
        else:
#                    print("这是没有拼接")
                   gcn_sem_dis = event_sem_input.unsqueeze(0)
#         gcn_inputs = torch.cat([gcn_inputs, event_sem_output], dim = -1)
        it = 1
        pre = gcn_inputs
        gcn_inputs = self.norm(gcn_inputs) + self.norm(gcn_sem_dis)
        ##无迭代
        gcn_layer = self.gcn_layers[0]
        gcn_outputs, adj_str = gcn_layer(gcn_inputs,pad_event1_pos, pad_event2_pos ,rel_type, event_pairs,node_event)
#         penal +=  adj_str.size(0) / torch.norm(adj_sem + adj_str)
        ##迭代
#         while iteration < max_iteration:
#             ##GCN_input
#             gcn_layer = self.gcn_layers[int(iteration / 2)]
#             gcn_outputs, adj_str = gcn_layer(gcn_inputs,pad_event1_pos, pad_event2_pos ,rel_type, event_pairs,node_event)
#             if self.iscat == 1:
#                    penal +=  adj_str.size(0) / torch.norm(adj_sem + adj_str)
#             iteration += 1
#             it = it + 1
#             gcn_inputs = self.norm(gcn_outputs) + 1 / it * self.norm(gcn_sem_dis)
        event_str_input = gcn_outputs
        event_str_input = event_str_input.squeeze(0)
        if self.iscat == 1:
                        event1_sem = torch.index_select(event_sem_output, 0, pad_event1_pos[0])
                        event2_sem = torch.index_select(event_sem_output, 0, pad_event2_pos[0])
        else:
                        event1_sem = torch.index_select(event_sem_input, 0, pad_event1_pos[0])
                        event2_sem = torch.index_select(event_sem_input, 0, pad_event2_pos[0])                        
        event1_str = torch.index_select(event_str_input, 0, pad_event1_pos[0])
        event2_str = torch.index_select(event_str_input, 0, pad_event2_pos[0])
                ###拼接transformer层
        event1 = torch.cat([event1_sem, event1_str], dim= -1)
        event2 = torch.cat([event2_sem, event2_str], dim= -1)
        event_pair_embed = torch.cat([event1, event2], dim=-1)
        event_diff = event2 - event1
        event_pair_pre = torch.cat((event_pair_embed, event_diff), dim=1)
        prediction = self.mlp(event_pair_pre)
        if self.loss_type == 'focal':
            loss += self.focal_loss(prediction, target) + self.penal1 * penal / max_iteration#0.7持平
        else:
            loss += self.ce_loss(prediction, target) + penal / max_iteration
        return loss, prediction

    # Extract the representation vector of the event
    def extract_event(self, embed, event_pid):
        e_1 = int(event_pid[0])
        e_2 = int(event_pid[1])
        e1_embed = torch.zeros(1, embedding_size).to(self.device)
        length = e_2 - e_1
        for j in range(e_1,e_2):
            e1_embed += embed[j]
        event_embed = e1_embed / length
        event_embed = self.liner_event(torch.cat((event_embed, embed[0].unsqueeze(0)),dim=-1))
        return event_embed


class bertCSRModel(nn.Module):
    def __init__(self, args):
        super(bertCSRModel, self).__init__()
        self.device = args.device
        self.pretrained_model = BertModel.from_pretrained(args.model_name)
        if args.iscat == 1:
                    self.mlp = nn.Sequential(nn.Linear((1 * args.n_last + 2 * embedding_size) * 2, args.mlp_size),
                                  nn.ReLU(), nn.Dropout(args.mlp_drop),
                                  nn.Linear(args.mlp_size, args.no_of_classes))
        else:
                    self.mlp = nn.Sequential(nn.Linear((1 * args.n_last + embedding_size) * 2, args.mlp_size),
                                  nn.ReLU(), nn.Dropout(args.mlp_drop),
                                  nn.Linear(args.mlp_size, args.no_of_classes))
        self.gcn_layers = GCN(768, 768, args.GCN_layers, args.GCN_heads)
#         for i in range(int(args.max_iteration / 2)):
#             self.gcn_layers.append(GCN(768, 768, args.GCN_layers, args.GCN_heads))
        # self.mlp_str = nn.Sequential(nn.Linear((1 * args.n_last + 2 * embedding_size), args.mlp_size),
        #                           nn.ReLU(), nn.Dropout(args.mlp_drop),
        #                           nn.Linear(args.mlp_size, args.no_of_classes))
        self.rate = args.rate
        self.w = args.w
        self.max_iteration = args.max_iteration
        self.threshold = args.threshold
        self.norm = nn.LayerNorm(embedding_size)
        self.min_iteration = args.min_iteration
        self.transformer_encoder = TransformerModel(768,768,args.Trans_head,args.Trans_layers,3072,512,0.1)##6 12达到最佳 drop 0.2 0.1
        self.liner_sem = nn.Linear(768, 768)
        self.liner_str = nn.Linear(768, 768)
        self.liner_event = nn.Linear(768 * 2, 768)
        self.penal1 = args.penal
        self.focal_loss = focal_loss(gamma=args.gamma, num_classes=args.no_of_classes)
        self.ce_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss_type = args.loss_type
        self.iscat = args.iscat

    def forward(self, enc_input_ids, enc_mask_ids, node_event, t1_pos, t2_pos, target, rel_type, event_pairs):
        sent_emb = self.pretrained_model(enc_input_ids, enc_mask_ids)[0].to(self.device)
        sent_list = []
        for idx, event in enumerate(node_event):
            sent_id = int(event[0])
            event_pid = event[1]
            e_emb = self.extract_event(sent_emb[sent_id], event_pid).to(self.device)
            if idx == 0:
                event_embed = e_emb
            else:
                event_embed = torch.cat((event_embed, e_emb))
            if sent_id not in sent_list:
                sent_list.append(sent_id)
        if len(event_pairs) > 1:
            target = torch.cat([torch.tensor(t) for t in target], dim=0).to(self.device)
        else:
            target = torch.tensor(target[0]).to(self.device)
#         if self.max_iteration > sent_list.__len__():
#             max_iteration = sent_list.__len__()
#         else:
#             max_iteration = self.max_iteration
        max_iteration = self.max_iteration
        ##未添加relu
        event_sem_input = self.liner_sem(event_embed)
        event_sem_input = F.relu(event_sem_input)
        event_str_input = self.liner_str(event_embed)
        # event_str_input = self.CGE.proj['event'](event_embed)
        event_str_input = F.relu(event_str_input)
        event_sem_output, adj_sem = self.transformer_encoder(event_sem_input)
        adj_sem[0] -= torch.diag(torch.diag(adj_sem[0]))
        adj_sem[0] += torch.eye(adj_sem[0].size(0)).cuda()
        pad_event1_pos = pad_sequence([torch.tensor(pos) for pos in t1_pos]).t().to(self.device).long()
        pad_event2_pos = pad_sequence([torch.tensor(pos) for pos in t2_pos]).t().to(self.device).long()

        iteration = 0
        difference = self.threshold
        loss = 0.0
        penal = 0.0
        gcn_inputs = event_str_input.unsqueeze(0)
        gcn_sem_dis = event_sem_output.unsqueeze(0)
#         gcn_inputs = torch.cat([gcn_inputs, event_sem_output], dim = -1)
        it = 1
        pre = gcn_inputs
        gcn_inputs = gcn_inputs
        gcn_layer = self.gcn_layers
        gcn_outputs, adj_str = gcn_layer(gcn_inputs)
#         penal += adj_str.size(0) / torch.norm(adj_sem + adj_str)
        while iteration < max_iteration:
            ##GCN_input
            gcn_layer = self.gcn_layers[int(iteration / 2)]
            gcn_outputs, adj_str = gcn_layer(gcn_inputs)
            penal += adj_str.size(0) / torch.norm(adj_sem + adj_str)
            # if iteration > 0:
            #     difference = self.Contrast_pre(adj_str, adj_str_1)
            iteration += 1
#             gcn_inputs = torch.cat([gcn_inputs, event_sem_output], dim = -1)
            it = it + 1
            if it == 2:
                gcn_inputs = gcn_outputs
#                 gcn_inputs = self.norm(gcn_outputs) + self.norm(pre)
                pre = gcn_outputs
            else:
                gcn_inputs = gcn_outputs
        event_str_input = gcn_outputs
        event_str_input = event_str_input.squeeze(0)
        event1_sem = torch.index_select(event_sem_output, 0, pad_event1_pos[0])
        event2_sem = torch.index_select(event_sem_output, 0, pad_event2_pos[0])
        event1_str = torch.index_select(event_str_input, 0, pad_event1_pos[0])
        event2_str = torch.index_select(event_str_input, 0, pad_event2_pos[0])
        ###拼接transformer层
        if self.iscat == 1:
                      print("这是二者")
                      event1 = torch.cat([event1_sem, event1_str], dim= -1)
                      event2 = torch.cat([event2_sem, event2_str], dim= -1)
        else:
                      print("这是只有结构")
                      event1 = event1_str
                      event2 = event2_str                    
        ##不拼接
        event1 = torch.cat([event1_sem, event1_str], dim= -1)
        event2 = torch.cat([event2_sem, event2_str], dim= -1)
        event_pair_embed = torch.cat([event1, event2], dim=-1)
        event_diff = event2 - event1
        event_pair_pre = torch.cat((event_pair_embed, event_diff), dim=1)
        prediction = self.mlp(event_pair_pre)
        if self.loss_type == 'focal':
            loss += self.focal_loss(prediction, target) + self.penal1 * penal #0.7持平
        else:
            loss += self.ce_loss(prediction, target) + penal
        return loss, prediction

    # Extract the representation vector of the event
    def extract_event(self, embed, event_pid):
        e_1 = int(event_pid[0])
        e_2 = int(event_pid[1])
        e1_embed = torch.zeros(1, embedding_size).to(self.device)
        length = e_2 - e_1
        for j in range(e_1,e_2):
            e1_embed += embed[j]
        event_embed = e1_embed / length
        event_embed = self.liner_event(torch.cat((event_embed, embed[0].unsqueeze(0)),dim=-1))
        return event_embed

    # Construct adjacency matrix
    def get_graphedge_index(self, prediction, event_pair, rel_type):
        graphedge_index = {}
        if self.training:
            rate = 0
        else:
            rate = self.w
        if rate != 0:
            pred_soft = torch.softmax(prediction, dim=1)
            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 1 and rel_type[0][j] == 0)]).t().to(self.device)
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 2 and rel_type[0][j] == 0)]).t().to(self.device)
            if tmp.size()[0] > 0:
                intra_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 1 and rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                intra_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 2 and rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_2 = tmp_1
            intra_graphedge_index = torch.cat((intra_graphedge_index_1, intra_graphedge_index_2), dim=-1)

            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 1 and rel_type[0][j] == 1)]).t().to(self.device)
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 2 and rel_type[0][j] == 1)]).t().to(self.device)
            if tmp.size()[0] > 0:
                inter_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 1 and rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                inter_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (torch.max(pred_soft[j], dim=0)[0] > rate and torch.max(pred_soft[j], dim=0)[1] == 2 and rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_2 = tmp_1
            inter_graphedge_index = torch.cat((inter_graphedge_index_1, inter_graphedge_index_2), dim=-1)
        else:
            predt = torch.argmax(prediction, dim=1)
            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (predt[j] == 1 and rel_type[0][j] == 0)]).t().to(self.device)
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (predt[j] == 2 and rel_type[0][j] == 0)]).t().to(self.device)
            if tmp.size()[0] > 0:
                intra_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (predt[j] == 1 and rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                intra_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (predt[j] == 2 and rel_type[0][j] == 0)]), dim=-1)
            else:
                intra_graphedge_index_2 = tmp_1
            intra_graphedge_index = torch.cat((intra_graphedge_index_1, intra_graphedge_index_2), dim=-1)

            tmp = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (predt[j] == 1 and rel_type[0][j] == 1)]).t().to(self.device)
            tmp_1 = torch.tensor([[pair[0], pair[1]] for j, pair in enumerate(event_pair) if (predt[j] == 2 and rel_type[0][j] == 1)]).t().to(self.device)
            if tmp.size()[0] > 0:
                inter_graphedge_index_1 = torch.cat(([torch.cat((torch.tensor([pair[0]]).unsqueeze(0).to(self.device), torch.tensor([pair[1]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (predt[j] == 1 and rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_1 = tmp
            if tmp_1.size()[0] > 0:
                inter_graphedge_index_2 = torch.cat(([torch.cat((torch.tensor([pair[1]]).unsqueeze(0).to(self.device), torch.tensor([pair[0]]).unsqueeze(0).to(self.device)), dim=0) for j, pair in enumerate(event_pair) if (predt[j] == 2 and rel_type[0][j] == 1)]), dim=-1)
            else:
                inter_graphedge_index_2 = tmp_1
            inter_graphedge_index = torch.cat((inter_graphedge_index_1, inter_graphedge_index_2), dim=-1)
        if intra_graphedge_index.size()[0] > 0:
            graphedge_index[('event', 'intra', 'event')] = intra_graphedge_index.long()
        if inter_graphedge_index.size()[0] > 0:
            graphedge_index[('event', 'inter', 'event')] = inter_graphedge_index.long()
        return graphedge_index

    # Compare the structural differences before and after iteration
    def Contrast_pre(self, prediction, prediction_last):
        if self.training:
            rate = self.rate
        else:
            rate = self.w
        if rate != 0:
            pred_soft = torch.softmax(prediction, dim=1)
            pred_last_soft = torch.softmax(prediction_last, dim=1)
            pre_list = torch.tensor([]).to(self.device)
            pre_last_list = torch.tensor([]).to(self.device)
            max_pro, pred_t = torch.max(pred_soft, dim=1)
            for idx, pre in enumerate(max_pro):
                if pre > self.rate:
                    pre_list = torch.cat((pre_list, pred_t[[idx]]), dim=-1)
                else:
                    pre_list = torch.cat((pre_list, torch.tensor([0]).to(self.device)), dim=-1)
            max_pro, pred_t = torch.max(pred_last_soft, dim=1)
            for idx, pre in enumerate(max_pro):
                if pre > self.rate:
                    pre_last_list = torch.cat((pre_last_list, pred_t[[idx]]), dim=-1)
                else:
                    pre_last_list = torch.cat((pre_last_list, torch.tensor([0]).to(self.device)), dim=-1)
        else:
            pre_list = torch.argmax(prediction, dim=1)
            pre_last_list = torch.argmax(prediction_last, dim=1)
        different = (pre_list != pre_last_list).sum().item()
        return different

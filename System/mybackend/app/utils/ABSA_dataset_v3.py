
import pickle

import torch
import numpy as np

from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence


def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens
class ABSA_Dataset(Dataset):
    def __init__(self, path):
        super(ABSA_Dataset, self).__init__()
        data = path
        self.bert_tokens = [torch.LongTensor(bert_token) for bert_token in data['bert_sequence_list']]
        self.bert_N_S_sequence_list = [torch.LongTensor(bert_token) for bert_token in data['bert_N_S_sequence_list']]
        self.bert_sub_sequence_list = [torch.LongTensor(bert_token) for bert_token in data['bert_sub_sequence_list']]
        self.aspect_masks = [torch.LongTensor(bert_mask) for bert_mask in data['aspect_nodeindex_list']]
        self.labels = torch.LongTensor(data['label_list'])
        self.len = len(data['label_list'])
        self.length = torch.LongTensor(data['length'])
        self.bert_length = torch.LongTensor(data['bert_length'])
        self.bert_N_S_length_list = torch.LongTensor(data['bert_N_S_length_list'])
        self.bert_sub_length_list = torch.LongTensor(data['bert_sub_length_list'])
        self.rel_adj = data['rel_adj_list']
        self.node_num = torch.LongTensor(data['node_num_list'])
        self.sub_adj_list = data['sub_adj_list']
        self.bert_segments_ids = [torch.LongTensor(bert_token) for bert_token in data['bert_segments_ids_list']]
        self.bert_N_S_segments_ids_list = [torch.LongTensor(bert_token) for bert_token in data['bert_N_S_segments_ids_list']]
        self.bert_sub_segments_ids_list = [torch.LongTensor(bert_token) for bert_token in data['bert_sub_segments_ids_list']]
        self.word_mapback =[torch.LongTensor(bert_token) for bert_token in data['word_mapback']]
        self.word_mapback_N_S_list = [torch.LongTensor(bert_token) for bert_token in data['word_mapback_N_S_list']]
        self.word_mapback_sub_list = [torch.LongTensor(bert_token) for bert_token in data['word_mapback_sub_list']]
        self.sub_rel_adj = data['sub_rel_adj_list']
        self.sub_graph_ind_list = data['sub_graph_ind_list']
        self.adj_list = data['adj_list']
        self.rel_adj_mask_list = data['rel_adj_mask_list']
        self.aspect_in_sub_list = [torch.LongTensor(bert_token) for bert_token in data['aspect_in_sub_list']]
        self.adj_matrix_pos_list = [torch.LongTensor(bert_token) for bert_token in data['adj_matrix_pos_list']]
        self.all_aspect = data['all_aspect']
        self.all_aspect_label = data['all_aspect_label']
        self.all_aspect_label_reverse = data['all_aspect_label_reverse']
        self.all_aspect_label_a = data['all_aspect_label_a']
        self.aspect_num_list = data['aspect_num_list']
        self.all_clause = data['all_clause']
        self.all_clause_label = data['all_clause_label']
        self.all_clause_re_label = data['all_clause_re_label']
        self.clause_num = data['sub_length_list']
        self.clause_label_super_label_list = data['clause_label_super_label_list']
        self.clause_label_super_reverse_label_list = data['clause_label_super_reverse_label_list']
    def __getitem__(self, index):
        return (self.length[index],
                self.bert_length[index],
                self.word_mapback[index],
                self.sub_adj_list[index],
                self.rel_adj[index],
                self.node_num[index],
                self.bert_segments_ids[index],
                self.word_mapback_N_S_list[index],
                self.word_mapback_sub_list[index],
                self.sub_rel_adj[index],
                self.aspect_masks[index],
                self.bert_tokens[index],
                self.labels[index],
                self.bert_N_S_sequence_list[index],
                self.bert_sub_sequence_list[index],
                self.bert_N_S_length_list[index],
                self.bert_sub_length_list[index],
                self.bert_N_S_segments_ids_list[index],
                self.bert_sub_segments_ids_list[index],
                self.sub_graph_ind_list[index],
                self.adj_list[index],
                self.rel_adj_mask_list[index],
                self.aspect_in_sub_list[index],
                self.adj_matrix_pos_list[index],
                self.all_aspect[index],
                self.all_aspect_label[index],
                self.aspect_num_list[index],
                self.all_aspect_label_reverse[index],
                self.all_aspect_label_a[index],
                self.all_clause[index],
                self.all_clause_label[index],
                self.clause_num[index],
                self.all_clause_re_label[index],
                self.clause_label_super_label_list[index],
                self.clause_label_super_reverse_label_list[index]
                #self.rel_list[index]
                )

    def __len__(self):
        return self.len


def ABSA_collate_fn(batch):
    batch_size = len(batch)
    batch = list(zip(*batch))

    lens = batch[0]
    lens1 = batch[5]
    (length_, bert_length_, word_mapback_,
     sub_adj_list_, rel_adj_,
     node_num_, bert_segments_ids_list_,
     word_mapback_N_S_list_, word_mapback_sub_list_,
     sub_rel_adj_,
     aspect_masks_,bert_tokens_,labels_,bert_N_S_sequence_list_,bert_sub_sequence_list_,bert_N_S_length_list,bert_sub_length_list,bert_N_S_segments_ids_list,
     bert_sub_segments_ids_list,sub_graph_ind_list,adj_list,rel_adj_mask_list,aspect_in_sub_list,adj_matrix_pos_list,all_aspect,all_aspect_label,aspect_num_list,all_aspect_label_reverse,all_aspect_label_a,all_clause,all_clause_label,clause_num,all_clause_re_label,clause_label_super_label_list,clause_label_super_reverse_label_list) = batch
    #rel_list_1 = pad_sequence(rel_list,batch_first=True)
    # list2 = [101, 1045, 2106, 19948, 2041, 1996, 2524, 3298, 2005, 1037, 19102, 6640, 2692, 7020, 2094, 2029, 1045,
    #          3811, 16755, 1012, 102, 1044, 1037, 1054, 1040, 1040, 1054, 1045, 1058, 1041, 102]
    # list2 = torch.tensor(list2)
    # if list2 in bert_tokens_:
    #     print(1)
    aspect_in_sub_list = pad_sequence(aspect_in_sub_list,batch_first=True)
    bert_sub_segments_ids_list = pad_sequence(bert_sub_segments_ids_list,batch_first=True)
    bert_N_S_segments_ids_list = pad_sequence(bert_N_S_segments_ids_list, batch_first=True)
    bert_tokens = pad_sequence(bert_tokens_,batch_first=True)
    aspect_masks = pad_sequence(aspect_masks_,batch_first=True)
    bert_N_S_sequence_list = pad_sequence(bert_N_S_sequence_list_,batch_first=True)
    bert_sub_sequence_list = pad_sequence(bert_sub_sequence_list_,batch_first=True)
    max_lens_se = max(lens)#最大句子长度
    max_lens_su = max(lens1)##最大子图数
    max_lens_cl = max(clause_num)
    max_aspect_num = max(aspect_num_list)##最大方面数
    word_mapback = pad_sequence(word_mapback_, batch_first=True)
    bert_segments_ids_list = pad_sequence(bert_segments_ids_list_,batch_first=True)
    word_mapback_N_S_list = pad_sequence(word_mapback_N_S_list_, batch_first=True)
    word_mapback_sub_list = pad_sequence(word_mapback_sub_list_, batch_first=True)
    all_aspect_label_adj_aa = np.zeros((batch_size, max_aspect_num, max_aspect_num),
                           dtype=np.float32)
    all_aspect_label_adj_reverse = np.zeros((batch_size, max_aspect_num, max_aspect_num),
                           dtype=np.float32)
    all_aspect_label_adj = np.zeros((batch_size, max_aspect_num, max_aspect_num),
                           dtype=np.float32)
    all_clause_label_adj = np.zeros((batch_size, max_aspect_num, max_aspect_num),
                           dtype=np.float32)
    all_clause_re_label_adj = np.zeros((batch_size, max_aspect_num, max_aspect_num),
                           dtype=np.float32)
    all_clause_super_label_adj = np.zeros((batch_size, max_lens_cl, max_lens_cl),
                           dtype=np.float32)
    all_clause_super_re_label_adj = np.zeros((batch_size, max_lens_cl, max_lens_cl),
                           dtype=np.float32)
    for idx in range(batch_size):
        mlen1 = all_aspect_label_reverse[idx].shape[0]
        mlen2 = all_aspect_label_reverse[idx].shape[1]
        all_aspect_label_adj_reverse[idx,:mlen1,:mlen2] = all_aspect_label_reverse[idx]
        all_aspect_label_adj[idx,:mlen1,:mlen2] = all_aspect_label[idx]
        all_aspect_label_adj_aa[idx, :mlen1, :mlen2] = all_aspect_label_a[idx]
        all_clause_label_adj[idx, :mlen1, :mlen2] = all_clause_label[idx]
        all_clause_re_label_adj[idx, :mlen1, :mlen2] = all_clause_re_label[idx]
    all_clause_adj = np.zeros((batch_size, max_aspect_num, max_lens_cl),
                           dtype=np.float32)
    for idx in range(batch_size):
        mlen1 = all_clause[idx].shape[0]
        mlen2 = all_clause[idx].shape[1]
        mlen3 = clause_label_super_label_list[idx].shape[0]
        mlen4 = clause_label_super_reverse_label_list[idx].shape[1]
        all_clause_adj[idx,:mlen1,:mlen2] = all_clause[idx]
        all_clause_super_label_adj[idx,:mlen3,:mlen4] = clause_label_super_label_list[idx]
        all_clause_super_re_label_adj[idx,:mlen3,:mlen4] = clause_label_super_reverse_label_list[idx]
    all_aspect_adj = np.zeros((batch_size, max_aspect_num, max_lens_se),
                           dtype=np.float32)
    for idx in range(batch_size):
        mlen1 = all_aspect[idx].shape[0]
        mlen2 = all_aspect[idx].shape[1]
        all_aspect_adj[idx,:mlen1,:mlen2] = all_aspect[idx]
    adj_oneshot5 = pad_sequence(adj_matrix_pos_list, batch_first=True)

    adj_oneshot4 = np.zeros((batch_size, max_lens_se, max_lens_se),
                           dtype=np.float32)
    for idx in range(batch_size):
        mlen1 = rel_adj_mask_list[idx].shape[0]
        adj_oneshot4[idx,:mlen1,:mlen1] = rel_adj_mask_list[idx]
    adj_oneshot3 = np.zeros((batch_size, max_lens_su, max_lens_se),
                           dtype=np.float32)
    for idx in range(batch_size):
        mlen1 = sub_graph_ind_list[idx].shape[0]
        mlen2 = sub_graph_ind_list[idx].shape[1]
        adj_oneshot3[idx,:mlen1,:mlen2] = sub_graph_ind_list[idx]
    adj_oneshot2 = np.zeros((batch_size, (max_lens_su), max_lens_su),
                           dtype=np.float32)
    for idx in range(batch_size):
        mlen1 = sub_rel_adj_[idx].shape[0]
        adj_oneshot2[idx,:mlen1,:mlen1] = sub_rel_adj_[idx]
    adj_oneshot1 = np.zeros((batch_size, max_lens_se, max_lens_se),
                           dtype=np.float32)
    for idx in range(batch_size):
        mlen1 = rel_adj_[idx].shape[0]
        adj_oneshot1[idx,:mlen1,:mlen1] = rel_adj_[idx]

    adj_oneshot = np.zeros((batch_size, max_lens_se, max_lens_se), dtype=np.float32)

    for idx in range(batch_size):
        mlen = adj_list[idx].shape[0]
        adj_oneshot[idx, :mlen, :mlen] = adj_list[idx]
    adj_oneshot = torch.FloatTensor(adj_oneshot)##信息流邻接矩阵BxNxN
    adj_oneshot1 = torch.LongTensor(adj_oneshot1)##依赖关系矩阵BxNxN
    adj_oneshot2 = torch.FloatTensor(adj_oneshot2)##子图关系邻接矩阵Bxnodexnode子图关系矩阵的掩码
    adj_oneshot3 = torch.FloatTensor(adj_oneshot3)##子图在原句子中的mask矩阵BxnodexN用于获取子句表征##aspect_in_sub_list为方面词在哪个子句中Bxnode
    adj_oneshot4 = torch.FloatTensor(adj_oneshot4)##依赖关系矩阵的mask矩阵BxNxN
    #adj_oneshot5 = torch.FloatTensor(adj_oneshot5)##pos的关系矩阵
    all_aspect_adj = torch.FloatTensor(all_aspect_adj)##所有方面词的索引
    all_aspect_label_adj = torch.FloatTensor(all_aspect_label_adj)##负例mask矩阵
    all_aspect_label_adj_reverse = torch.FloatTensor(all_aspect_label_adj_reverse)##正例mask矩阵
    all_aspect_label_adj_aa = torch.FloatTensor(all_aspect_label_adj_aa)
    all_clause_adj = torch.FloatTensor(all_clause_adj)
    all_clause_label_adj = torch.FloatTensor(all_clause_label_adj)
    all_clause_re_label_adj = torch.FloatTensor(all_clause_re_label_adj)
    all_clause_super_label_adj = torch.FloatTensor(all_clause_super_label_adj)
    all_clause_super_re_label_adj = torch.FloatTensor(all_clause_super_re_label_adj)
    length_ = torch.stack(length_)
    bert_length_ = torch.stack(bert_length_)
    labels_ = torch.stack(labels_)
    bert_N_S_length_list = torch.stack(bert_N_S_length_list)
    bert_sub_length_list = torch.stack(bert_sub_length_list)
    return (
        length_, bert_length_, word_mapback, adj_oneshot,
        adj_oneshot1, adj_oneshot2,
        bert_segments_ids_list, word_mapback_N_S_list,
        word_mapback_sub_list, bert_sub_sequence_list,
        bert_N_S_sequence_list, aspect_masks,
        bert_tokens,adj_oneshot3,max_lens_su,bert_N_S_length_list,bert_sub_length_list,
        bert_N_S_segments_ids_list,bert_sub_segments_ids_list,adj_oneshot4,aspect_in_sub_list,adj_oneshot5,all_aspect_adj,all_aspect_label_adj,all_aspect_label_adj_reverse,all_aspect_label_adj_aa,all_clause_adj,all_clause_label_adj,all_clause_re_label_adj,all_clause_super_label_adj,all_clause_super_re_label_adj,labels_
    )
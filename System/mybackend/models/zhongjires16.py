import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, BertConfig
import torch.nn.functional as F
import numpy as np
class ScaledDotProductAttention(nn.Module):
    """ Scaled Dot-Product Attention """

    def __init__(self, scale):
        super().__init__()

        self.scale = scale
        self.softmax = nn.Softmax(dim=2)
        self.q_linear = nn.Linear(768,256)
        self.k_linear = nn.Linear(768,256)
        self.v_linear = nn.Linear(768,768)
    def forward(self, q, k, v, mask=None):
        q = self.q_linear(q)
        k = self.k_linear(k)
        u = torch.bmm(q, k.transpose(1, 2)) # 1.Matmul
        u = u / self.scale # 2.Scale

        if mask is not None:
            u = u.masked_fill(mask, -np.inf) # 3.Mask
        temperature = 1e-14
        # attn = self.softmax(u/temperature) # 4.Softmax
        output = torch.bmm(u, v) # 5.Output

        return u, output
def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size,  seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*apply_permutation.*")
warnings.filterwarnings("ignore", message=".*attention_mask.*")
class MCDG(nn.Module):
    _keys_to_ignore_on_load_unexpected = [r"pooler", 'cls.predictions.bias',
                                          'cls.predictions.transform.dense.weight',
                                          'cls.predictions.transform.dense.bias',
                                          'cls.predictions.decoder.weight',
                                          'cls.seq_relationship.weight',
                                          'cls.seq_relationship.bias',
                                          'cls.predictions.transform.LayerNorm.weight',
                                          'cls.predictions.transform.LayerNorm.bias']
    _keys_to_ignore_on_load_missing = [r"position_ids", r"decoder.bias", r"classifier",
                                       'cls.bias', 'cls.transform.dense.weight',
                                       'cls.transform.dense.bias', 'cls.transform.LayerNorm.weight',
                                       'cls.transform.LayerNorm.bias', 'cls.decoder.weight',
                                       'cls_representation.weight', 'cls_representation.bias',
                                       'aspect_representation.weight', 'aspect_representation.bias']

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = Bert_Encoder(args)
        self.classifier = nn.Linear(512,3)
        self.dropout = nn.Dropout(0.1)
        self.Word_GNN = Word_GNN(args,48)
        self.Clause_GNN = Clause_GNN(args)
        self.sub_rel_linear = nn.Linear(768,768)
        self.linear1 = nn.Linear(256,256)
        self.linear2 = nn.Linear(256, 256)
        self.temperature = 0.07
        self.A = args.A
        self.B = args.B
        # self.dropout1 = nn.Dropout(0.1)
        # self.att = SelfAttention(n_head=4,d_x=768,d_k=128,d_v=128,d_o=256)
        # self.rel_linear = nn.Linear(768, 256)
        # self.pos_linear = nn.Linear(256, 1)
    def forward(self, inputs,concat=False,Sum=True,dot=False):
        length_, bert_length_, word_mapback, adj_oneshot,\
            adj_oneshot1, adj_oneshot2,\
            bert_segments_ids_list, word_mapback_N_S_list,\
            word_mapback_sub_list, bert_sub_sequence_list,\
            bert_N_S_sequence_list, aspect_masks,\
            bert_tokens,adj_oneshot3,max_node_num,bert_N_S_length_list,bert_sub_length_list,bert_N_S_segments_ids_list,bert_sub_segments_ids_list,adj_oneshot4,aspect_in_sub_list,adj_oneshot5,all_aspect_adj,all_aspect_label_adj,all_aspect_label_adj_reverse,all_aspect_label_adj_aa,all_clause_adj,all_clause_label_adj,all_clause_re_label_adj,all_clause_super_label_adj,all_clause_super_re_label_adj,label = inputs
        # bert_encoder
        # list1 = list(bert_tokens)
        # list2 = [101, 1045, 2106, 19948, 2041, 1996, 2524, 3298, 2005, 1037, 19102, 6640, 2692, 7020, 2094, 2029, 1045, 3811, 16755, 1012, 102, 1044, 1037, 1054, 1040, 1040, 1054, 1045, 1058, 1041, 102]
        # if len(list1[0]) > len(list2):
        #     list_temp = [0] * (len(list1[0]) - len(list2))
        #     list2 = list2 + list_temp
        # list2 = torch.tensor(list2).cuda()
        # for i in list1:
        #     if torch.equal(i, list2):
        #         print(1)
        encoder_input = (bert_length_, word_mapback,bert_segments_ids_list,bert_tokens)
        encoder_input1 = (bert_N_S_length_list, word_mapback_N_S_list, bert_N_S_segments_ids_list, bert_N_S_sequence_list)
        encoder_input2 = (bert_sub_length_list, word_mapback_sub_list, bert_sub_segments_ids_list, bert_sub_sequence_list)
        word_embedding = self.encoder(encoder_input)
        B = word_embedding.shape[0]
        N = word_embedding.shape[1]
        # att_adj,output = self.att(word_embedding)
        # att_adj = att_adj.view(4,B,N,N)
        # att_adj_sum = torch.sum(att_adj,dim=0) / 4
        N_S_embedding = self.encoder(encoder_input1)
        sub_rel_embedding = self.encoder(encoder_input2)

        if concat:
            merge_N_S_sub_rel = torch.concat((N_S_embedding, sub_rel_embedding), dim=-1)
            merge_N_S_sub_rel = F.sigmoid(merge_N_S_sub_rel)
            merge_N_S_sub_rel = self.sub_rel_linear(merge_N_S_sub_rel)
            merge_N_S_sub_rel = self.dropout(F.relu(merge_N_S_sub_rel))
        elif Sum:
            merge_N_S_sub_rel = N_S_embedding + sub_rel_embedding
            # merge_N_S_sub_rel = F.sigmoid(merge_N_S_sub_rel)
            # merge_N_S_sub_rel = self.dropout(merge_N_S_sub_rel)
            # merge_N_S_sub_rel = F.relu(merge_N_S_sub_rel)
            merge_N_S_sub_rel = self.sub_rel_linear(merge_N_S_sub_rel)
            merge_N_S_sub_rel = self.dropout(F.relu(merge_N_S_sub_rel))
        else:
            merge_N_S_sub_rel = N_S_embedding * sub_rel_embedding
            merge_N_S_sub_rel = self.sub_rel_linear(merge_N_S_sub_rel)
            merge_N_S_sub_rel = self.dropout(F.relu(merge_N_S_sub_rel))

        # merge_N_S_sub_rel = self.dropout(F.relu(merge_N_S_sub_rel))
        Word_GNN_output = self.Word_GNN(adj_oneshot, adj_oneshot1, word_embedding,adj_oneshot4,adj_oneshot5,Dot=True,Sum=False,Concat=False)
        aspect_masks = aspect_masks.unsqueeze(1).to(dtype=torch.float32)
        raw_aspect = torch.bmm(aspect_masks,word_embedding)
        aspect_num = torch.sum(aspect_masks, dim=-1)
        aspect_num.masked_fill_(aspect_num == 0, 1)
        raw_aspect_emmbedding = raw_aspect / aspect_num.unsqueeze(dim=-1)
        raw_aspect_emmbedding = raw_aspect_emmbedding.repeat(1,all_clause_adj.shape[2],1)
        # raw_aspect_emmbedding = self.rel_linear(raw_aspect_emmbedding)
        # word_embedding_aaa = torch.sum(word_embedding,dim=-1)
        # word_embedding_aaa.masked_fill_(word_embedding_aaa != 0, 1)
        # raw_aspect_emmbedding = raw_aspect_emmbedding * word_embedding_aaa.unsqueeze(-1)
        # word_embedding_1 = word_embedding * raw_aspect_emmbedding
        # word_embedding_1 = word_embedding+raw_aspect_emmbedding
        # word_embedding_1 = torch.concat((word_embedding,raw_aspect_emmbedding),dim=-1)
        Clause_GNN_output,wnt1 = self.Clause_GNN(word_embedding,adj_oneshot2,merge_N_S_sub_rel,adj_oneshot3) #adj_oneshot3为获取子图表征
        Clause_GNN_output_norm = F.normalize(Clause_GNN_output,dim=-1)
        Clause_GNN_output_sim = torch.bmm(Clause_GNN_output_norm,Clause_GNN_output_norm.permute(0, 2, 1))
        Clause_GNN_output_sim_1 = (Clause_GNN_output_sim + 1) / 2

        Clause_GNN_output_sim_loss = self.sim_nnnn(Clause_GNN_output_sim_1,all_clause_super_label_adj,all_clause_super_re_label_adj)
        # Clause_GNN_output = Clause_GNN_output * (raw_aspect_emmbedding + Clause_GNN_output)
        # Clause_GNN_output = F.sigmoid(Clause_GNN_output)


        # aspect_word = self.linear1(aspect_word)
        # ############################
        # aspect_word_all = self.linear1(Word_GNN_output)
        # aspect_num = torch.sum(all_aspect_adj, dim=-1)
        # aspect_num.masked_fill_(aspect_num == 0, 1)
        # all_aspect_emmbedding = torch.bmm(all_aspect_adj, aspect_word_all) / aspect_num.unsqueeze(dim=-1)
        # aspect_word_all_1 = F.normalize(all_aspect_emmbedding, dim=-1)
        #
        # all_aspect_cos_sim = torch.bmm(aspect_word_all_1, aspect_word_all_1.permute(0, 2, 1))
        # all_aspect_cos_sim = (all_aspect_cos_sim + 1)/2
        # sim_loss_word = self.sim_lllll(all_aspect_cos_sim, all_aspect_label_adj,all_aspect_label_adj_reverse)
        # ###########################
        # aspect_clause = torch.bmm(all_clause_adj,Clause_GNN_output)
        # wnt = all_clause_adj.sum(dim=-1)
        # wnt.masked_fill_(wnt == 0, 1)
        # aspect_clause = aspect_clause / wnt.unsqueeze(dim=-1)
        # aspect_clause = self.linear2(aspect_clause)
        # aspect_clause_all_1 = F.normalize(aspect_clause, dim=-1)
        #
        # all_clause_cos_sim = torch.bmm(aspect_clause_all_1, aspect_clause_all_1.permute(0, 2, 1))
        # all_clause_cos_sim = (all_clause_cos_sim + 1)/2
        # sim_loss_clause = self.sim_c(all_clause_cos_sim, all_aspect_label_adj)
        # sim_loss_end = sim_loss_word + sim_loss_clause
        # aspect_clause = aspect_clause[:,0,:].unsqueeze(1)
        ##################

        ##################
        # raw_aspect_emmbedding = self.rel_linear(word_embedding)
        aspect_word_all = self.linear1(Word_GNN_output)
        aspect_num = torch.sum(all_aspect_adj, dim=-1)
        aspect_num.masked_fill_(aspect_num == 0, 1)
        all_aspect_emmbedding = torch.bmm(all_aspect_adj, aspect_word_all) / aspect_num.unsqueeze(dim=-1)
        # all_aspect_emmbedding_temp = all_aspect_emmbedding[:,0,:].unsqueeze(1)
        # all_aspect_emmbedding_temp =all_aspect_emmbedding_temp.repeat(1,Clause_GNN_output.shape[1],1)
        # Clause_GNN_output = Clause_GNN_output * F.sigmoid((all_aspect_emmbedding_temp + Clause_GNN_output))
        # Clause_GNN_output = F.sigmoid(Clause_GNN_output)

        # output = F.sigmoid(output)
        # output = self.dropout(output)
        # raw_aspect_emmbedding = torch.bmm(all_aspect_adj, output) / aspect_num.unsqueeze(dim=-1)
        # raw_aspect_emmbedding = self.rel_linear(raw_aspect_emmbedding)
        # raw_aspect_emmbedding = raw_aspect_emmbedding.repeat(1,all_aspect_adj.shape[1],1)
        # raw_aspect_emmbedding = F.normalize(raw_aspect_emmbedding,dim=-1)
        # aspect_word_all = self.linear1(Word_GNN_output)
        # aspect_num = torch.sum(all_aspect_adj, dim=-1)
        # aspect_num.masked_fill_(aspect_num == 0, 1)
        # all_aspect_emmbedding = torch.bmm(all_aspect_adj, aspect_word_all) / aspect_num.unsqueeze(dim=-1)
        aspect_word_all_1 = F.normalize(all_aspect_emmbedding, dim=-1)
        cos_sim = torch.bmm(aspect_word_all_1, aspect_word_all_1.permute(0, 2, 1))
        cos_sim = (cos_sim + 1)/2
        loss_word = self.sim_lllll(cos_sim,all_aspect_label_adj,all_aspect_label_adj_reverse)

        aspect_clause = torch.bmm(all_clause_adj,Clause_GNN_output)
        wnt = all_clause_adj.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        aspect_clause = aspect_clause / wnt.unsqueeze(dim=-1)
        # aspect_clause = torch.concat((aspect_clause, raw_aspect_emmbedding), dim=-1)
        aspect_clause = self.linear2(aspect_clause)
        # aspect_clause = aspect_clause * raw_aspect_emmbedding

        # aspect_masks = aspect_masks.unsqueeze(1).to(dtype=torch.float32)
        # raw_aspect = torch.bmm(aspect_masks,word_embedding)
        # aspect_num = torch.sum(aspect_masks, dim=-1)
        # aspect_num.masked_fill_(aspect_num == 0, 1)
        # raw_aspect_emmbedding = raw_aspect / aspect_num.unsqueeze(dim=-1)
        # raw_aspect_emmbedding = self.rel_linear(raw_aspect_emmbedding)
        # raw_aspect_a = self.pos_linear(raw_aspect_emmbedding * aspect_clause)
        # raw_aspect_emmbedding = F.sigmoid(raw_aspect_a)



        # clause_em = self.linear2(Clause_GNN_output)
        # clause_em = clause_em * wnt1
        # clause_em_temp = F.normalize(clause_em,dim=-1)
        ################
        # aspect_word = torch.bmm(aspect_masks.unsqueeze(1).to(dtype=torch.float32),word_embedding)
        # wnt = aspect_masks.sum(dim=-1)
        # wnt.masked_fill_(wnt == 0, 1)
        # raw_aspect = aspect_word / wnt.unsqueeze(dim=-1).unsqueeze(dim=-1)
        # num_clause = Clause_GNN_output.shape[1]
        # raw_aspect = raw_aspect.repeat(1,num_clause,1)
        # raw_aspect = self.rel_linear(raw_aspect)
        # raw_aspect_temp = F.normalize(raw_aspect,dim=-1)
        # raw_aspect_clause_score = torch.bmm(raw_aspect_temp,clause_em_temp.permute(0, 2, 1))
        # raw_aspect_clause_score.masked_fill_(raw_aspect_clause_score == 0, -1e30)
        # raw_aspect_clause_score = F.softmax(raw_aspect_clause_score,dim=-1)
        # raw_aspect_clause_ = torch.bmm(raw_aspect_clause_score,Clause_GNN_output)
        # raw_aspect_clause_ = raw_aspect_clause_[:,0,:].unsqueeze(1)
        # raw_aspect_clause_ = raw_aspect_clause_.repeat(1,aspect_clause.shape[1],1)
        #############
        aspect_clause_all_1 = F.normalize(aspect_clause, dim=-1)
        all_clause_cos_sim = torch.bmm(aspect_clause_all_1, aspect_clause_all_1.permute(0, 2, 1))
        # aspect_clause = torch.concat((aspect_clause,raw_aspect_emmbedding),dim=-1)
        # all_clause_cos_sim = (all_clause_cos_sim + 1) / 2
        # loss_clause = self.sim_lllll(all_clause_cos_sim,all_clause_label_adj,all_clause_re_label_adj)
        # aspect_clause_1 = torch.bmm(aspect_clause_all_1 ,raw_aspect_emmbedding.permute(0, 2, 1))[:,0,0]
        # aspect_clause = aspect_clause * aspect_clause_1.unsqueeze(1).unsqueeze(2)
        # all_aspect_emmbedding = all_aspect_emmbedding + raw_aspect_emmbedding
        #all_aspect_emmbedding = self.rel_linear(all_aspect_emmbedding)
        # aspect_clause = aspect_clause * raw_aspect_emmbedding
        aspect_Clause = torch.concat((all_aspect_emmbedding, aspect_clause), dim=2)
        # aspect_Clause = torch.concat((all_aspect_emmbedding, raw_aspect_clause_), dim=2)
        aspect_word_all_1 = F.normalize(aspect_Clause, dim=-1)
        all_aspect_cos_sim = torch.bmm(aspect_word_all_1, aspect_word_all_1.permute(0, 2, 1))
        all_aspect_cos_sim = (all_aspect_cos_sim + 1)/2
        sim_loss_end = self.A*Clause_GNN_output_sim_loss + self.B*loss_word
        # sim_loss_end = 0.8*Clause_GNN_output_sim_loss
        # sim_loss_end = 0
        # sim_loss_end = 0.8 * loss_word
        aspect_Clause = aspect_Clause[:,0,:].unsqueeze(1)
        aspect_Clause = self.dropout(aspect_Clause)
        logits = self.classifier(aspect_Clause)
        return logits,sim_loss_end,aspect_clause[:,0,:]
    def sim_nnnn(self,all_aspect_cos_sim,all_aspect_label_adj,all_aspect_label_adj_reverse):
        all_mask = torch.sum(torch.sum(all_aspect_label_adj, dim=-1), dim=-1)
        all_mask_add_ep = torch.add(all_mask,1e-8)
        all_mask_1 = torch.sum(torch.sum(all_aspect_label_adj_reverse, dim=-1), dim=-1)
        all_mask_add_ep_1 = torch.add(all_mask_1,1e-8)
        all_mask_all = all_mask + all_mask_1
        num_has_op_aspect_1 = len(torch.nonzero(all_mask))
        num_has_op_aspect_2 = len(torch.nonzero(all_mask_1))
        num_has_op_aspect_all = len(torch.nonzero(all_mask_all))
        ae = torch.sum(all_aspect_label_adj, dim=-1)
        ae_1 = torch.add(ae,1e-8)
        # ae_2 = torch.sum(ae!=0,dim=1)
        ab = torch.sum(all_aspect_label_adj_reverse, dim=-1)
        ab_1 = torch.add(ab,1e-8)
        # ab_2 = torch.sum(ab != 0, dim=1)
        if num_has_op_aspect_all == 0:
            return 0
        ########
        logits_max, _ = torch.max(all_aspect_cos_sim, dim=-1, keepdim=True)
        all_aspect_cos_sim_temp = (logits_max.detach() - all_aspect_cos_sim) * all_aspect_label_adj_reverse
        # exp_score_1 = (all_aspect_cos_sim_temp*all_aspect_label_adj_reverse)[:, 0, :]
        exp_score_1 = (torch.exp(all_aspect_cos_sim_temp)*all_aspect_label_adj_reverse)
        sum_exp_score_1 = torch.sum(exp_score_1,dim=-1)
        div_sum_exp_score_1 = torch.div(sum_exp_score_1,ab_1)
        sum_score_1 = torch.sum(div_sum_exp_score_1,dim=-1)
        sum_score_1 = torch.div(sum_score_1,all_mask_add_ep_1)
        sum_score_1 = torch.sum(sum_score_1)
        if num_has_op_aspect_2 == 0:
            log_sum_score_1 = torch.tensor(0,dtype=torch.float32).cuda()
        else:
            log_sum_score_1 = torch.log(sum_score_1) / num_has_op_aspect_2
        # log_sum_score_1 = sum_score_1 / num_has_op_aspect_2
        ########
        # exp_score = (all_aspect_cos_sim*all_aspect_label_adj)[:, 0, :]
        exp_score = (torch.exp(all_aspect_cos_sim)*all_aspect_label_adj)
        sum_exp_score = torch.sum(exp_score,dim=-1)
        div_sum_exp_score = torch.div(sum_exp_score,ae_1)
        sum_score = torch.sum(div_sum_exp_score,dim=-1)
        sum_score = torch.div(sum_score,all_mask_add_ep)
        sum_score = torch.sum(sum_score)
        if num_has_op_aspect_1 == 0:
            log_sum_score = torch.tensor(0,dtype=torch.float32).cuda()
        else:
            log_sum_score = torch.log(sum_score) / num_has_op_aspect_1
        # log_sum_score = sum_score / num_has_op_aspect_1
        # log_sum_score = torch.log(sum_score) / num_has_op_aspect_1
        only_ = 0.5*log_sum_score_1 + 0.5*log_sum_score
        # only_ = log_sum_score
        return only_
    def sim_c(self,all_aspect_cos_sim,all_aspect_label_adj):
        all_mask = torch.sum(torch.sum(all_aspect_label_adj, dim=-1), dim=-1)
        all_mask_add_ep = torch.add(all_mask,1e-8)
        num_has_op_aspect_1 = len(torch.nonzero(all_mask))
        if num_has_op_aspect_1 == 0:
            return 0
        exp_score = (torch.exp(all_aspect_cos_sim)*all_aspect_label_adj)[:,0,:]
        sum_exp_score = torch.sum(exp_score,dim=-1)
        div_sum_exp_score = torch.div(sum_exp_score,all_mask_add_ep)
        sum_score = torch.sum(div_sum_exp_score)
        if num_has_op_aspect_1 == 0:
            log_sum_score = torch.tensor(0,dtype=torch.float32).cuda()
        else:
            log_sum_score = torch.log(sum_score) / num_has_op_aspect_1
        only_ = log_sum_score
        return only_
    def sim_lllll(self,all_aspect_cos_sim,all_aspect_label_adj,all_aspect_label_adj_reverse):
        all_mask = torch.sum(torch.sum(all_aspect_label_adj, dim=-1), dim=-1)
        all_mask_add_ep = torch.add(all_mask,1e-8)
        all_mask_1 = torch.sum(torch.sum(all_aspect_label_adj_reverse, dim=-1), dim=-1)
        all_mask_add_ep_1 = torch.add(all_mask_1,1e-8)
        all_mask_all = all_mask + all_mask_1
        num_has_op_aspect_1 = len(torch.nonzero(all_mask))
        num_has_op_aspect_2 = len(torch.nonzero(all_mask_1))
        num_has_op_aspect_all = len(torch.nonzero(all_mask_all))
        if num_has_op_aspect_all == 0:
            return 0
        ########
        logits_max, _ = torch.max(all_aspect_cos_sim, dim=-1, keepdim=True)
        all_aspect_cos_sim_temp = (logits_max.detach() - all_aspect_cos_sim) * all_aspect_label_adj_reverse
        # exp_score_1 = (all_aspect_cos_sim_temp*all_aspect_label_adj_reverse)[:, 0, :]
        exp_score_1 = (torch.exp(all_aspect_cos_sim_temp)*all_aspect_label_adj_reverse)[:, 0, :]
        sum_exp_score_1 = torch.sum(exp_score_1,dim=-1)
        div_sum_exp_score_1 = torch.div(sum_exp_score_1,all_mask_add_ep_1)
        sum_score_1 = torch.sum(div_sum_exp_score_1)
        # sum_score_1[torch.nonzero(sum_score_1 == 0)] = 1
        if num_has_op_aspect_2 == 0:
            log_sum_score_1 = torch.tensor(0,dtype=torch.float32).cuda()
        else:
            log_sum_score_1 = torch.log(sum_score_1) / num_has_op_aspect_2
        # log_sum_score_1 = sum_score_1 / num_has_op_aspect_2
        ########
        # exp_score = (all_aspect_cos_sim*all_aspect_label_adj)[:, 0, :]
        exp_score = (torch.exp(all_aspect_cos_sim)*all_aspect_label_adj)[:,0,:]
        sum_exp_score = torch.sum(exp_score,dim=-1)
        div_sum_exp_score = torch.div(sum_exp_score,all_mask_add_ep)
        sum_score = torch.sum(div_sum_exp_score)
        if num_has_op_aspect_1 == 0:
            log_sum_score = torch.tensor(0,dtype=torch.float32).cuda()
        else:
            log_sum_score = torch.log(sum_score) / num_has_op_aspect_1
        # log_sum_score = sum_score / num_has_op_aspect_1
        # log_sum_score = torch.log(sum_score) / num_has_op_aspect_1
        only_ = 0.5*log_sum_score_1 + 0.5*log_sum_score
        # only_ = log_sum_score
        return only_
class Clause_GNN(nn.Module):
    def __init__(self, args, input_size=768, hidden_size=256, num_layers=2):
        super(Clause_GNN, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.gat_dep1 = Rel_GAT(args)
        self.gat_dep2 = Rel_GAT(args)
        self.gat_dep3 = Rel_GAT(args)
        self.gat_dep4 = Rel_GAT(args)
        self.gat_dep5 = Rel_GAT(args)
        self.gat_dep6 = Rel_GAT(args)
        # self.gat_dep7 = Rel_GAT(args)
        # self.gat_dep8 = Rel_GAT(args)
        self.gat_dep = [self.gat_dep1,self.gat_dep2,self.gat_dep3,self.gat_dep4,self.gat_dep5,self.gat_dep6]
        self.merge = nn.Linear(6*512,256)

    def forward(self,gru_input,rel_mask,rel_adj,adj_oneshot3):
        gru_output,_ = self.gru(gru_input)
        gru_output = torch.bmm(adj_oneshot3,gru_output)
        wnt = adj_oneshot3.sum(dim=-1)
        wnt1 = adj_oneshot3.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        wnt1.masked_fill_(wnt1 != 0, 1)
        gru_output = gru_output / wnt.unsqueeze(dim=-1)
        feature = [g(rel_mask,rel_adj,gru_output) for g in self.gat_dep]
        feature = torch.cat(feature, dim=2)
        feature_out = self.merge(feature)
        # wnt1 = wnt1.unsqueeze(dim=-1)
        # feature_out = feature_out * wnt1
        return feature_out,wnt1
class Word_GNN(nn.Module):
    """
    Relation gat model, use the embedding of the edges to predict attention weight
    """

    def __init__(self, args, dep_rel_num, hidden_size=64, num_layers=2):
        super(Word_GNN, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.1)
        # self.leakyrelu = nn.LeakyReLU(1e-2)
        self.linear = nn.Linear(768,768)
        # self.linear1 = nn.Linear(768, 256)
        self.linear3 = nn.Linear(768, 256)
        self.dep_rel_embed = nn.Embedding(
            dep_rel_num, args.dep_relation_embed_dim)
        nn.init.xavier_uniform_(self.dep_rel_embed.weight, gain=5)
        self.a = nn.Linear(args.dep_relation_embed_dim, hidden_size)
        #self.b = nn.BatchNorm1d(hidden_size)
        self.b = nn.LayerNorm(hidden_size)
        self.c = nn.LeakyReLU(1e-2)
        self.d = nn.Linear(hidden_size, 1)
        self.f = nn.Linear(768,768)
        self.pos_rel_embed = nn.Embedding(
            dep_rel_num, args.dep_relation_embed_dim)
        nn.init.xavier_uniform_(self.pos_rel_embed.weight, gain=5)
        self.q = nn.Linear(768,768)
        self.k = nn.Linear(768,768)
        # self.linear1 = nn.Linear(768, 768)
        # self.linear2 = nn.Linear(768, 768)
    def forward(self, adj, rel_adj, feature,rel_mask,adj_oneshot5,att_adj_sum=None,Dot=False,Sum=False,Concat=False):
        B, N = rel_mask.size(0), rel_mask.size(1)
        sem_feature = F.normalize(feature,dim=-1)
        sem_adj = torch.bmm(sem_feature,sem_feature.permute(0, 2, 1)) * rel_mask
        sem_adj = sem_adj.view(B,-1)
        # sem_adj = att_adj_sum.view(B,-1)
        rel_adj_V = self.dep_rel_embed(
            rel_adj.view(B, -1))  # (batch_size, n*n, d)
        pos_adj = self.pos_rel_embed(adj_oneshot5)
        pos_adj = pos_adj.unsqueeze(1).repeat(1,N,1,1).permute(0, 2, 1, 3)
        dmask = rel_mask  # (batch_size, n*n)
        a1 = torch.unsqueeze(torch.eye(N), dim=0).expand(B, -1, -1).to('cuda')
        # gcn layer
        # relation based GAT, attention over relations
        rel_adj_V_reverse = ((rel_mask - a1) * rel_mask).permute(0, 2, 1).unsqueeze(-1) * (rel_adj_V.reshape(B,N,N,768).permute(0, 2, 1,3))
        mask_reverse = ((rel_mask - a1) * rel_mask).permute(0, 2, 1)
        cc = a1 * adj
        for i in range(2):
            if Dot:
                rel_adj_V_temp = (rel_adj_V.view(B, N, N, 768) * (dmask.unsqueeze(-1)))
                cf = feature.repeat(1, N, 1)# 让所有0位置的元素为0
                tt = self.f(cf)
                tt = tt.reshape(B, N, N, 768)
                # feature_temp = torch.sigmoid(tt + rel_adj_V_temp)######81.97
                # rel_adj_V_temp = (pos_adj * feature_temp)
                feature_temp = torch.sigmoid(tt + pos_adj)######81.5
                rel_adj_V_temp = (rel_adj_V_temp * feature_temp)
                rel_adj_V_temp = rel_adj_V_temp.reshape(B,-1,768)
                #rel_adj_V_temp = rel_adj_V_temp + F.sigmoid(self.f(pos_adj.reshape(B,-1,768)))
            if Sum:
                rel_adj_V_temp = (rel_adj_V.view(B, N, N, 768) * (dmask.unsqueeze(-1)))  # 让所有0位置的元素为0
                feature_temp = F.sigmoid((feature.repeat(1, N, 1))).reshape(B, N, N, 768).permute(0, 2, 1, 3)
                rel_adj_V_temp = (rel_adj_V_temp + feature_temp).reshape(B,-1,768)
            if Concat:
                rel_adj_V_temp = (rel_adj_V.view(B, N, N, 768) * (dmask.unsqueeze(-1)))  # 让所有0位置的元素为0
                feature_temp = (feature.repeat(1, N, 1)).reshape(B, N, N, 768).permute(0, 2, 1, 3)
                rel_adj_V_temp = torch.concat((rel_adj_V_temp,feature_temp),dim=-1).reshape(B,-1,2*768)
            a = self.a(rel_adj_V_temp)
            b = self.b(a.view(-1,64)).view(B,-1,64) if a.view(-1,64).shape[0] != 1 else a
            c = self.c(b)
            rel_adj_logits = self.d(c).squeeze(2)
            mask_lo = mask_logits(rel_adj_logits + sem_adj, dmask.view(B,-1)).reshape(B,N,N)
            rel_adj_logits1 = F.softmax(
                mask_lo,
                dim=2)
            # rel_adj_logits = F.sigmoid(rel_adj_logits)
            # cc = (rel_adj_logits * (dmask - a1).view(B,-1)).reshape(B,N,N)
            # cc_sum = torch.sum(cc,dim=2)
            # cc_sum = torch.where(cc_sum == 0, torch.tensor(1.0), cc_sum)
            # diag_values = torch.diag_embed(cc_sum)
            # cc *= ~torch.eye(N, dtype=torch.bool).unsqueeze(0).cuda()  # 将对角线上的元素设为0
            # cc += diag_values
            # rel_adj_logits1 = F.softmax(
            #     mask_logits(cc,dmask),
            #     dim=2)
            rel_feature = torch.bmm(rel_adj_logits1,feature)
            rel_feature_temp = rel_feature.repeat(1, N, 1)
            temp1 = self.q(rel_feature_temp)
            temp1 = temp1.reshape(B, N, N, 768)
            temp2 = self.k(rel_adj_V_reverse)
            temp2 = temp2.reshape(B, N, N, 768)
            # ak = (temp1 + temp2) * pos_adj.permute(0, 2, 1, 3)
            ak = temp1 + temp1.permute(0, 2, 1, 3) + temp2
            # ak = (temp1 + temp1.permute(0, 2, 1, 3) + temp2) * pos_adj.permute(0, 2, 1, 3)
            # ak = ((temp1*temp1.permute(0, 2, 1, 3) + pos_adj*pos_adj.permute(0, 2, 1, 3)) + temp2)
            feature_temp = torch.sigmoid(ak)
            feature_temp1 = rel_feature.repeat(1, N, 1).reshape(B, N, N, 768)
            feature_temp1 = feature_temp1.permute(0, 2, 1, 3)
            feature_temp1 = feature_temp1 * cc.unsqueeze(-1)
            # feature_temp1 = (rel_feature.repeat(1, N, 1).reshape(B, N, N, 768).permute(0, 2, 1, 3) * cc.unsqueeze(-1))
            feature_temp2 = rel_feature.repeat(1, N, 1).reshape(B, N, N, 768)
            feature_temp2 = feature_temp2 * mask_reverse.unsqueeze(-1)
            feature_temp2 = feature_temp2 * (1 - feature_temp)
            # feature_temp2 = (rel_feature.repeat(1, N, 1).reshape(B, N, N, 768) * mask_reverse.unsqueeze(-1)) * (1 - feature_temp)
            feature_temp = torch.sum(feature_temp1 + feature_temp2,dim=2)

            adj_out = feature_temp

            if i == 0:
                adj_out = self.linear(adj_out)
                adj_out = F.relu(adj_out)
                feature = self.dropout(adj_out)
            # elif i == 1:
            #     adj_out = self.linear1(adj_out)
            #     adj_out = F.relu(adj_out)
            #     feature = self.dropout(adj_out)
            # elif i ==2 :
            #     adj_out = self.linear2(adj_out)
            #     adj_out = F.relu(adj_out)
            #     feature = self.dropout(adj_out)
            else:
                feature = self.linear3(adj_out)
        return feature
def mask_logits(target, mask):
    return target * mask + (1 - mask) * (-1e30)


class Rel_GAT(nn.Module):
    """
    Relation gat model, use the embedding of the edges to predict attention weight
    """

    def __init__(self, args, hidden_size=512, num_layers=2):
        super(Rel_GAT, self).__init__()
        self.args = args
        self.num_layers = num_layers
        self.dropout = nn.Dropout(0.2)
        # self.leakyrelu = nn.LeakyReLU(1e-2)
        self.a = nn.Linear(args.dep_relation_embed_dim, hidden_size)
        # self.b = nn.BatchNorm1d(hidden_size)
        self.b = nn.LayerNorm(hidden_size)
        self.c = nn.LeakyReLU(1e-2)####nn.ReLU()且层数为4准去率也是81
        self.d = nn.Linear(hidden_size, 1)
        self.f = nn.Linear(hidden_size,args.dep_relation_embed_dim)
        # self.linear1 = nn.Linear(512,512)
        # self.linear2 = nn.Linear(512,512)

    def forward(self, adj, rel_adj, feature):
        denom = adj.sum(2).unsqueeze(2) + 1
        B, N = adj.size(0), adj.size(1)
        # sem_feature_q = self.linear1(feature)
        # sem_feature_q = F.normalize(sem_feature_q,dim=-1)
        # sem_feature_k = self.linear1(feature)
        # sem_feature_k = F.normalize(feature,dim=-1)
        sem_adj = torch.bmm(feature,feature.permute(0, 2, 1)) * adj
        sem_adj = sem_adj.view(B,-1)
        Dot = True
        # gcn layer
        #adj = adj + torch.unsqueeze(torch.eye(N), dim=0).expand(B, -1, -1).to('cuda')
        for l in range(self.num_layers):
            # relation based GAT, attention over relations
            ###new
            # feature2 = feature.unsqueeze(1).repeat(1, N, 1, 1)
            # feature_temp = self.linear1(feature2.reshape(B, -1, 512))
            # feature_temp = s  elf.dropout(F.relu(feature_temp))
            # rel_adj_temp = rel_adj + feature_temp
            if Dot:
                rel_adj_V_temp = (rel_adj.view(B, N, N, 768) * (adj.unsqueeze(-1)))  # 让所有0位置的元素为0
                tt = self.f(feature.repeat(1, N, 1))
                tt = tt.reshape(B, N, N, 768)
                tt = tt + tt.permute(0, 2, 1, 3)
                # feature_temp = self.dropout(F.relu(tt))
                feature_temp = torch.sigmoid(tt)
                rel_adj_V_temp = (rel_adj_V_temp * feature_temp).reshape(B,-1,768)
            else:
                rel_adj_V_temp = (rel_adj.view(B, N, N, 768) * (adj.unsqueeze(-1)))  # 让所有0位置的元素为0
                tt = self.f(feature.repeat(1, N, 1))
                tt = tt.reshape(B, N, N, 768)
                tt = tt + tt.permute(0, 2, 1, 3)
                feature_temp = self.dropout(F.relu(tt))
                rel_adj_V_temp = (rel_adj_V_temp + feature_temp).reshape(B,-1,768)
            a = self.a(rel_adj_V_temp)
            b = self.b(a.view(-1,512)).view(B,-1,512) if a.view(-1,512).shape[0] != 1 else a
            c = self.c(b)
            rel_adj_logits = self.d(c).squeeze(2)
            #rel_adj_logits = F.sigmoid(rel_adj_logits)
            ###
            # rel_adj_logits = self.fcs(rel_adj).squeeze(2)  # (batch_size, n*n)
            ####BN
            #
            ####
            dmask = adj.view(B, -1)  # (batch_size, n*n)
            # rel_adj_logits = F.softmax(
            #     mask_logits(rel_adj_logits + sem_adj, dmask).reshape(B,N,N), dim=2)
            mask_lo = mask_logits(rel_adj_logits+torch.unsqueeze(torch.eye(N), dim=0).expand(B, -1, -1).to('cuda').reshape(B,-1),dmask).reshape(B,N,N)
            # mask_lo = mask_logits(rel_adj_logits + sem_adj,dmask).reshape(B,N,N)
            rel_adj_logits = F.softmax(mask_lo,dim=2)
            # rel_adj_logits = rel_adj_logits.view(
            #     *rel_adj.size())  # (batch_size, n, n)
            # rel_adj_logits = (rel_adj_logits + attn_weights)/2
            Ax = rel_adj_logits.bmm(feature)
            # if l ==0:
            #     Ax = F.relu(self.linear1(Ax))
            # else:
            #     Ax = self.linear2(Ax)
            # feature = Ax
            feature = self.dropout(F.relu(Ax)) if l < self.num_layers - 1 else Ax

        return feature
class Bert_Encoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args

        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_config.output_hidden_states = True

        self.layer_drop = nn.Dropout(args.layer_dropout)
        self.context_encoder = BertModel.from_pretrained('bert-base-uncased', config=bert_config)
        self.dense = nn.Linear(args.bert_hidden_dim, args.hidden_dim)

    def forward(self, inputs):

        bert_length_, word_mapback,bert_segments_ids_list,bert_tokens = inputs

        ###############################################################
        # 1. contextual encoder
        bert_outputs = self.context_encoder(bert_tokens, token_type_ids=bert_segments_ids_list)

        bert_out, bert_pooler_out = bert_outputs.last_hidden_state, bert_outputs.pooler_output

        bert_out = self.layer_drop(bert_out)
        # rm [CLS]
        bert_seq_indi = ~sequence_mask(bert_length_).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(bert_length_) + 1, :] * bert_seq_indi.float()
        word_mapback_one_hot = (F.one_hot(word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))

        # average
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        bert_out = bert_out / wnt.unsqueeze(dim=-1)
        return bert_out
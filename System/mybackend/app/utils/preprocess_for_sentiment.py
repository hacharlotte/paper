import json
import pickle
from lxml import etree
from transformers import BertTokenizer
from tqdm import tqdm
from supar import Parser
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
sentiment_dict = {
    'positive': 0,
    'negative': 1,
    'neutral': 2,
}
def text2bert_id(token, tokenizer):
    re_token = []
    word_mapback = []
    word_split_len = []
    for idx, word in enumerate(token):
        temp = tokenizer.tokenize(word)
        re_token.extend(temp)
        word_mapback.extend([idx] * len(temp))
        word_split_len.append(len(temp))
    re_id = tokenizer.convert_tokens_to_ids(re_token)
    return re_id ,word_mapback, word_split_len
def parse_json(path, lowercase=True):
    dataset = []
    with open(path, 'rb') as f:
        f = json.load(f)
        dep_parser = Parser.load(r'D:\NLP\BiSyn_GAT_plus\BiSyn_GAT_plus\ptb.biaffine.dep.lstm.char')
        for sentence in f:
            dict1 = {}
            dict1['token'] = sentence['token']
            dict1['sub_start_end'] = sentence["sub_start_end"]
            dict1['sub_rel_list'] = sentence["sub_rel_list"]
            datase = dep_parser.predict(dict1['token'], verbose=False)
            dep_head = datase.arcs[0]
            dep_rel = datase.rels[0]
            dict1['dep_head'] = [x - 1 for x in dep_head]
            dict1['dep_rel'] = dep_rel
            dict1['pos_tag'] = nltk.pos_tag(sentence['token'])
            dataset.append(dict1)
    return dataset
def parse_xml(path, lowercase=False, remove_list=None):
    if remove_list is None:
        remove_list = []
    dataset = []
    with open(path, 'rb') as f:
        root = etree.fromstring(f.read())
        for sentence in root:
            index = sentence.get('id')
            sent = sentence.find('text').text
            if lowercase:
                sent = sent.lower()
            terms = sentence.find('aspectTerms')
            if terms is None:
                continue
            accept_terms = []
            for term in terms:
                aspect = term.attrib['term']
                sentiment = term.attrib['polarity']
                implicit = term.attrib.get('implicit_sentiment', '') == "True"
                if sentiment in remove_list:
                    continue
                left_index = int(term.attrib['from'])
                right_index = int(term.attrib['to'])
                left_word_offset = len(word_tokenize(sent[:left_index]))
                to_word_offset = len(word_tokenize(sent[:right_index]))
                if left_word_offset == to_word_offset:
                    left_word_offset = left_word_offset - 1
                accept_terms.append({
                    'aspect_term': aspect,
                    'sentiment': sentiment,
                    'implicit': implicit,
                    'left_index': left_word_offset,
                    'right_index': to_word_offset,
                })
            if accept_terms:
                dataset.append({
                    'id': index,
                    'text': sent,
                    'aspect_terms': accept_terms,
                })
    return dataset
def dataset_end(data1,data2):
    print(len(data1),len(data2))
    dataset = []
    for index,i in enumerate(data1):
        sent = i['text']
        token = word_tokenize(sent)
        if token != data2[index]['token']:
            print('1',token)
            print('2',data2[index]['token'])
        i.update(data2[index])
        dataset.append(i)
    return dataset
def head_to_adj_oneshot(heads, sent_len, aspect_dict,
                        leaf2root=True, root2leaf=True, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx.
    """
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)

    heads = heads[:sent_len]

    # aspect <self-loop>
    # for asp in aspect_dict:
    #     from_ = asp['left_index']
    #     to_ = asp['to']
    #     for i_idx in range(from_, to_):
    #         for j_idx in range(from_, to_):
    #             adj_matrix[i_idx][j_idx] = 1



    for idx, head in enumerate(heads):
        if head != -1:
            if leaf2root:
                adj_matrix[head, idx] = 0
            if root2leaf:
                adj_matrix[idx, head] = 1

        if self_loop:
            adj_matrix[idx, idx] = 1

    return adj_matrix
def pre_data():
    path = r'D:\NLP\RGAT-ABSA-master\RGAT-ABSA-master\data\semeval14\Restaurants_Train_v2.xml'
    path2 = r'DMRST_Parser-main\res_train.json'
    json_du = open('res_train_new.json', 'w', encoding='utf-8')
    dataset1 = parse_xml(path, lowercase=True)
    dataset2 = parse_json(path2)
    dataset = dataset_end(dataset1, dataset2)
    json.dump(dataset, json_du)
def pre_test_data():
    path = r'D:\NLP\RGAT-ABSA-master\RGAT-ABSA-master\data\semeval14\Laptops_Test_Gold.xml'
    path2 = r'DMRST_Parser-main\laptop_test.json'
    json_du = open('laptop_test_new.json', 'w', encoding='utf-8')
    dataset1 = parse_xml(path, lowercase=True)
    dataset2 = parse_json(path2)
    dataset = dataset_end(dataset1, dataset2)
    json.dump(dataset, json_du)

def hhhh(original_list):
    new_list = []
    temp_list = []

    # 遍历原始列表
    for item in original_list:
        if item == '0':
            # 遇到0时，将临时列表添加到新列表，并清空临时列表
            if temp_list:
                new_list.append(temp_list)
                temp_list = []
        else:
            # 将非0元素添加到临时列表
            temp_list.append(item)

    # 处理最后一个子列表（如果原始列表以非0元素结尾）
    if temp_list:
        new_list.append(temp_list)
    return new_list
def abcd(rels,heads,sub_end,rel_dict,edge_dict):
    sent_len = len(heads)
    cl_num = len(sub_end)
    adj_matrix = np.zeros((sent_len, cl_num), dtype=np.float32)
    keys_list = list(adict.keys())
    adj_matrix1 = np.zeros((sent_len, cl_num), dtype=np.float32)
    forward = np.zeros((sent_len, cl_num), dtype=np.float32)
    backward = np.zeros((sent_len, cl_num), dtype=np.float32)
    bb = []
    for aaa, sub_graphddd in enumerate(sub_end):
        b = list(range(sub_graphddd[0], sub_graphddd[1]))
        bb.append(b)
    aa = []
    for i in bb:
        thisway = []
        for j in i:
            if heads[j]!=-1:
                if heads[j] not in i:
                    if rels[j] == 'punct':
                        continue
                    left = bb.index(i) + 1
                    right = 0
                    for jk,jkd in enumerate(bb):
                        if heads[j] in jkd:
                            right = jk
                    right +=1
                    key_to_find = rel_dict[(left,right)]
                    position = keys_list.index(key_to_find)
                    adj_matrix[heads[j],bb.index(i)] = len(edge_dict) + position
                    forward[heads[j],bb.index(i)] = 1
                    adj_matrix1[j,right-1] = edge_dict[rels[j]]
                    backward[j,right-1] = 1
                    thisway.append((heads[j],j))
        if thisway!=[]:
            aa.append(thisway)
    return adj_matrix,adj_matrix1,forward,backward,aa
def rel_to_adj_oneshot(rels,heads, sent_len, aspect_dict,edge_dict,
                        leaf2root=True, root2leaf=True, self_loop=True):
    """
    Convert a sequence of head indexes into a 0/1 matirx.
    """
    adj_matrix = np.zeros((sent_len, sent_len), dtype=np.float32)

    heads = heads[:sent_len]

    for idx, head in enumerate(heads):
        if head != -1:
            if leaf2root:
                adj_matrix[head, idx] = edge_dict[rels[idx]]
            if root2leaf:
                adj_matrix[idx, head] = 0

        if self_loop:
            adj_matrix[idx, idx] = len(edge_dict)

    return adj_matrix

tokenizer = BertTokenizer.from_pretrained(r'D:\NLP\RGAT-ABSA-master\RGAT-ABSA-master\bert-base-uncased')
# print(dataset_end(dataset1,dataset2))
def rel_adj_deal_v1(created,node1,node2,rel_dict,N_S_dict):
    created.append((node1[0], node2[0]))
    created.append((node2[0], node1[0]))
    if node1[2] == node2[2]:
        rel1 = node1[3]
        rel2 = node1[3]
    elif node1[2] == "Satellite":
        rel2 = node1[3]
        rel1 = node2[3]
    else:
        rel2 = node2[3]
        rel1 = node1[3]
    rel_dict[(node1[0], node2[0])] = rel1
    rel_dict[(node2[0], node1[0])] = rel2
    if (node1[0], node2[0]) not in N_S_dict:
        N_S_dict[(node1[0], node2[0])] = node1[2]
        N_S_dict[(node2[0], node1[0])] = node2[2]
    return created,rel_dict,N_S_dict
def rel_adj_deal(created,node1,node2,rel_dict,N_S_dict):
    created.append((node1[0], node2[0]))
    created.append((node2[0], node1[0]))
    if node1[2] == node2[2]:
        rel = node1[3]
    elif node1[2] == "Satellite":
        rel = node1[3]
    else:
        rel = node2[3]
    rel_dict[(node1[0], node2[0])] = rel
    rel_dict[(node2[0], node1[0])] = rel
    if (node1[0], node2[0]) not in N_S_dict:
        N_S_dict[(node1[0], node2[0])] = node1[2]
        N_S_dict[(node2[0], node1[0])] = node2[2]
    return created,rel_dict,N_S_dict
def rel_adj_deal_1(created, node1, node2, rel_dict, N_S_dict):
    for aaa in range(node1[0], node1[1] + 1):
        created.append((aaa, node2[0]))
        created.append((node2[0], aaa))
        if node1[2] == node2[2]:
            rel = node1[3]
        elif node1[2] == "Satellite":
            rel = node1[3]
        else:
            rel = node2[3]
        rel_dict[(aaa, node2[0])] = rel
        rel_dict[(node2[0], aaa)] = rel
        if (aaa, node2[0]) not in N_S_dict:
            N_S_dict[(aaa, node2[0])] = node1[2]
            N_S_dict[(node2[0], aaa)] = node2[2]
    return created,rel_dict,N_S_dict
adict = {}
def process(data, tokenizer,edge_dict, pos_dict, lowercase=True):
    processed = []
    max_len = 100
    CLS_id = tokenizer.convert_tokens_to_ids(["[CLS]"])
    SEP_id = tokenizer.convert_tokens_to_ids(["[SEP]"])
    node_num_list = []
    length_list = []
    word_mapback_N_S_list = []
    word_mapback_sub_list = []
    bert_length_list = []
    bert_N_S_length_list = []
    bert_sub_length_list = []
    sub_length_list = []
    sub_adj_list = []
    bert_sequence_list = []
    bert_N_S_sequence_list = []
    bert_sub_lsequence_list = []
    bert_segments_ids_list = []
    label_list = []
    aspect_indi_list = []
    mapback_list = []
    relation_list = []
    sub_index_list = []
    sub_rel_adj_list = []
    rel_adj_list = []
    rel_adj_mask_list = []
    adj = []
    bert_N_S_segments_ids_list = []
    bert_sub_segments_ids_list = []
    sub_graph_ind_list = []
    aspect_in_sub_list = []
    adj_matrix_pos_list = []
    rel_list = []
    word_mapback_rel_list = []
    rel_adj_bert_length_list = []
    bert_rel_sequence_indices1_list = []
    bert_rel_segments_ids_list = []
    all_aspect = []
    all_clause = []
    all_clause_label = []
    all_clause_re_label = []
    all_aspect_label = []
    aspect_num_list = []
    all_aspect_label_reverse = []
    all_aspect_label_a = []
    clause_num = []
    clause_label_super_label_list = []
    clause_label_super_reverse_label_list = []
    for d in data:
        if d['text'] == "i did swap out the hard drive for a samsung 830 ssd which i highly recommend.":
            print(1)
        tok = d['token']
        pos_tag = d['pos_tag']
        if lowercase==True:
            tok = [t.lower() for t in tok]
        text_raw_bert_indices, word_mapback, _ = text2bert_id(tok, tokenizer)
        text_raw_bert_indices = text_raw_bert_indices[:max_len]
        word_mapback = word_mapback[:max_len]  # 将bertid与原单词映射
        length = word_mapback[-1] + 1  # 单词个数
        adj_matrix_pos = []
        for iii,pos_tag_i in enumerate(pos_tag):
            adj_matrix_pos.append(pos_dict[pos_tag_i[1]])

        bert_length = len(word_mapback)
        dep_head = d['dep_head'][:length]
        dep_rel = d['dep_rel']
        sub_graph_node = []
        sub_graph_node_new = []
        created = []
        sum1 = 1
        rel_dict = {}
        N_S_dict = {}
        if len(d["sub_start_end"])!=1:##这一步就是处理子句的关系，让其便于组织成矩阵形式
            my_list = d["sub_rel_list"]
            reversed_list = list(reversed(my_list))
            for node in reversed_list:
                if node[0][3] not in adict:
                    adict[node[0][3]] = 1
                else:
                    adict[node[0][3]] += 1
                if node[1][3] not in adict:
                    adict[node[1][3]] = 1
                else:
                    adict[node[1][3]] += 1
                if node[0][0]==node[0][1] and node[1][0]==node[1][1]:
                    created, rel_dict, N_S_dict = rel_adj_deal(created,node[0],node[1],rel_dict,N_S_dict)
                elif node[0][0]!=node[0][1] and node[1][0]==node[1][1]:
                    created, rel_dict, N_S_dict = rel_adj_deal_1(created,node[0],node[1],rel_dict,N_S_dict)
                elif node[0][0]==node[0][1] and node[1][0]!=node[1][1]:
                    created, rel_dict, N_S_dict = rel_adj_deal_1(created,node[1],node[0],rel_dict,N_S_dict)
                elif node[0][0]!=node[0][1] and node[1][0]!=node[1][1]:
                    list1 = range(node[0][0],node[0][1]+1)
                    list2 = range(node[1][0],node[1][1]+1)
                    paired_lists = [(x, y) for x in list1 for y in list2]
                    flag = False
                    for paired_lists_index in paired_lists:
                        if paired_lists_index in created:
                            flag = True
                    if flag:
                        print('进来了')
                        sum1 += 1
                        created = []
                        paired_lists_1 = [(x, y) for x in list2 for y in list1]
                        created = created+paired_lists_1+paired_lists
                    else:
                        paired_lists_1 = [(x, y) for x in list2 for y in list1]
                        created = created + paired_lists_1 + paired_lists
                        if node[0][2] == node[1][2]:
                            rel = node[0][3]
                        elif node[0][2] == "Satellite":
                            rel = node[0][3]
                        else:
                            rel = node[0][3]
                        for paired_index in paired_lists:
                            if paired_index not in rel_dict:
                                rel_dict[paired_index] = rel
                            if paired_index not in N_S_dict:
                                N_S_dict[paired_index] = node[0][2]
                        for paired_index in paired_lists_1:
                            if paired_index not in rel_dict:
                                rel_dict[paired_index] = rel
                            if paired_index not in N_S_dict:
                                N_S_dict[paired_index] = node[1][2]
                if (node[0][0],node[0][1]) not in sub_graph_node and node[0][0] == node[0][1]:
                    sub_graph_node.append((node[0][0],node[0][1]))
                elif (node[0][0],node[0][1]) not in sub_graph_node_new and node[0][0] != node[0][1]:
                    sub_graph_node_new.append((node[0][0],node[0][1]))
                if (node[1][0],node[1][1]) not in sub_graph_node and node[1][0] == node[1][1]:
                    sub_graph_node.append((node[1][0],node[1][1]))
                elif (node[1][0],node[1][1]) not in sub_graph_node_new and node[1][0] != node[1][1]:
                    sub_graph_node_new.append((node[1][0],node[1][1]))
                if (node[0][0],node[1][1]) not in sub_graph_node_new:
                    sub_graph_node_new.append((node[0][0],node[1][1]))
        sub_sequence = [] ##组成文本序列
        N_S_sequence = []
        adj_matrix = np.zeros((len(d["sub_start_end"]), len(d["sub_start_end"])), dtype=np.float32)
        len1 = len(d["sub_start_end"])
        if len(d["sub_rel_list"]) > 0:
            for leni in range(len1):
                for lenj in range(len1):
                    if (leni+1, lenj+1) in N_S_dict:
                        N_S_sequence.append(N_S_dict[(leni+1, lenj+1)])
                        adj_matrix[leni, lenj] = 1
                    else:
                        # N_S_sequence.append('[PAD]')
                        N_S_sequence.append('adjself')
                        adj_matrix[leni, lenj] = 1
            for leni in range(len1):
                for lenj in range(len1):
                    if (leni+1, lenj+1) in rel_dict:
                        sub_sequence.append(rel_dict[(leni+1, lenj+1)])
                    else:
                        # sub_sequence.append('[PAD]')
                        sub_sequence.append('adjself')
                        adj_matrix[leni, lenj] = 1
        N_S_raw_bert_indices, word_mapback_N_S, _ = text2bert_id(N_S_sequence, tokenizer)
        sub_raw_bert_indices, word_mapback_sub, _ = text2bert_id(sub_sequence, tokenizer)

        bert_N_S_length = len(word_mapback_N_S)
        bert_sub_length = len(word_mapback_sub)

        aspect_num = 0
        for aspect_1,aspect in enumerate(d['aspect_terms']):
            label = aspect['sentiment']
            if label not in sentiment_dict:
                continue
            aspect_num +=1
        clause_label_super_label = np.zeros((len(d["sub_start_end"]), len(d["sub_start_end"])), dtype=np.float32)
        clause_label_super_reverse_label = np.zeros((len(d["sub_start_end"]), len(d["sub_start_end"])), dtype=np.float32)
        if len(d["sub_start_end"]) != 1:
            for dict_key, dict_value in rel_dict.items():
                left1 = dict_key[0] -1
                right1 = dict_key[1] -1
                if dict_value == 'Contrast' or dict_value == 'Comparison':
                    clause_label_super_label[left1][right1] = 1
                    clause_label_super_reverse_label[left1][right1] = 0
                else:
                    clause_label_super_label[left1][right1] = 0
                    clause_label_super_reverse_label[left1][right1] = 1
        for aspect in d['aspect_terms']:
            label = aspect['sentiment']
            term = aspect['aspect_term']
            if label not in sentiment_dict:
                continue
            clause_label_super_label_list.append(clause_label_super_label)
            clause_label_super_reverse_label_list.append(clause_label_super_reverse_label)
            aspect_num_list.append(aspect_num)
            aspect_indi = np.zeros((aspect_num, length), dtype=np.float32)
            aspect_clause_indi = np.zeros((aspect_num, len(d['sub_start_end'])), dtype=np.float32)
            # aspect_clause_reverse = np.zeros((aspect_num, len(d['sub_start_end'])), dtype=np.float32)
            label_temp = []
            clause_label_temp = []
            clause_re_label_temp = []
            count = 0
            aspec_span_1 = list(range(aspect['left_index'], aspect['right_index']))
            flag_1 = 0
            flag_sum = 0
            a = [0] * len(d['sub_start_end'])
            for aaa,sub_graphddd in enumerate(d['sub_start_end']):
                b = list(range(sub_graphddd[0],sub_graphddd[1]))
                for aspec_span_index in aspec_span_1:
                    if aspec_span_index in b:
                        a[aaa] = 1
                        flag_1 = 1
                if flag_1 == 1:
                    flag_sum +=1
                    flag_1 =0
            aspect_in_sub_list.append(a)
            if flag_sum == 2:
                print(d['text'])
            for aspect_1, aspect_2 in enumerate(d['aspect_terms']):
                label1 = aspect_2['sentiment']
                if label1 not in sentiment_dict:
                    continue
                aspec_span = list(range(aspect_2['left_index'], aspect_2['right_index']))
                if aspect_2['aspect_term'] == term:
                    if len(label_temp) != 0:
                        temp1 = clause_re_label_temp[0]
                        clause_re_label_temp.append(temp1)
                        clause_re_label_temp[0] = 0
                        temp = clause_label_temp[0]
                        clause_label_temp.append(temp)
                        clause_label_temp[0] = 0
                        temp = aspect_clause_indi[0]
                        aspect_clause_indi[count] = temp
                        aspect_clause_indi[0] = [0] * len(aspect_clause_indi[0])
                        flag_1 = 0
                        flag_sum = 0
                        for aaa, sub_graphddd in enumerate(d['sub_start_end']):
                            b = list(range(sub_graphddd[0], sub_graphddd[1]))
                            for aspec_span_index in aspec_span:
                                if aspec_span_index in b:
                                    aspect_clause_indi[0][aaa] = 1
                                    flag_1 = 1
                            if flag_1 == 1:
                                flag_sum += 1
                                flag_1 = 0
                    else:
                        clause_re_label_temp.append(0)
                        clause_label_temp.append(0)
                        for aaa, sub_graphddd in enumerate(d['sub_start_end']):
                            b = list(range(sub_graphddd[0], sub_graphddd[1]))
                            for aspec_span_index in aspec_span:
                                if aspec_span_index in b:
                                    aspect_clause_indi[count][aaa] = 1
                    if len(label_temp) != 0:
                        temp = label_temp[0]
                        label_temp.append(temp)
                        label_temp[0] = 0
                        temp = aspect_indi[0]
                        aspect_indi[count] = temp
                        aspect_indi[0] = [0]*len(aspect_indi[0])
                        for pidx in range(aspect_2['left_index'], aspect_2['right_index']):
                            aspect_indi[0][pidx] = 1
                    else:
                        label_temp.append(0)
                        for pidx in range(aspect_2['left_index'], aspect_2['right_index']):
                            aspect_indi[count][pidx] = 1
                else:
                    for aaa, sub_graphddd in enumerate(d['sub_start_end']):
                        b = list(range(sub_graphddd[0], sub_graphddd[1]))
                        for aspec_span_index in aspec_span:
                            if aspec_span_index in b:
                                aspect_clause_indi[count][aaa] = 1
                    if label1 == label:
                        if a == list(aspect_clause_indi[count]):
                            clause_re_label_temp.append(0)
                        else:
                            clause_re_label_temp.append(1)
                        label_temp.append(0)
                        clause_label_temp.append(0)
                    else:
                        clause_re_label_temp.append(0)
                        if a == list(aspect_clause_indi[count]):
                            clause_label_temp.append(0)
                        else:
                            left = a.index(1) + 1
                            right = list(np.where(aspect_clause_indi[count]==1))
                            right =right[0].tolist()[0] + 1
                            if rel_dict[(left,right)]=='Contrast' or rel_dict[(left,right)]=='Comparison':
                                clause_label_temp.append(1)
                            else:
                                clause_label_temp.append(0)
                        label_temp.append(1)
                    for pidx in range(aspect_2['left_index'], aspect_2['right_index']):
                        aspect_indi[count][pidx] = 1
                count += 1

            all_clause.append(aspect_clause_indi)
            clause_label_temp = np.array(clause_label_temp,dtype=np.float32)
            clause_label_temp = clause_label_temp[np.newaxis , :]
            all_clause_label.append(clause_label_temp)

            clause_re_label_temp = np.array(clause_re_label_temp,dtype=np.float32)
            clause_re_label_temp = clause_re_label_temp[np.newaxis , :]
            all_clause_re_label.append(clause_re_label_temp)

            all_aspect.append(aspect_indi)
            label_temp = np.array(label_temp,dtype=np.float32)
            label_temp_1 = [0] * len(label_temp)
            label_temp_1[0] = 1
            label_temp_1 = np.array(label_temp_1,dtype=np.float32)
            label_temp_reverse = np.array([1]*len(label_temp),dtype=np.float32) - label_temp
            label_temp_reverse = label_temp_reverse - label_temp_1
            label_temp_1 = label_temp_1[np.newaxis , :]
            label_temp_reverse = label_temp_reverse[np.newaxis , :]
            label_temp = label_temp[np.newaxis , :]
            all_aspect_label.append(label_temp)
            all_aspect_label_reverse.append(label_temp_reverse)
            all_aspect_label_a.append(label_temp_1)
            rel_list_temp = [edge_dict[rel_list_i] for rel_list_i in dep_rel]
            rel_list.append(rel_list_temp)
            adj_matrix_pos_list.append(adj_matrix_pos)
            label = sentiment_dict[label]
            node_num_list.append(len(d["sub_start_end"]))
            sub_graph_ind_ = np.zeros((len(d["sub_start_end"]),length),dtype=np.float32)
            for sub_graph_ind_index,sub in enumerate(d["sub_start_end"]):
                sub_graph_ind_[sub_graph_ind_index,sub[0]:sub[1]] = np.ones((1,sub[1]-sub[0]),dtype=np.float32)
            sub_graph_ind_list.append(sub_graph_ind_)
            word_mapback_N_S_list.append(word_mapback_N_S)
            word_mapback_sub_list.append(word_mapback_sub)
            sub_rel_adj_list.append(adj_matrix)##子句邻接矩阵
            bert_N_S_length_list.append(bert_N_S_length)
            bert_sub_length_list.append(bert_sub_length)
            length_list.append(length)
            bert_length_list.append(bert_length)
            N_S_raw_bert_indices1 = CLS_id + N_S_raw_bert_indices + SEP_id##加上cls和sep
            sub_raw_bert_indices1 = CLS_id + sub_raw_bert_indices + SEP_id
            bert_N_S_sequence_list.append(N_S_raw_bert_indices1)
            bert_sub_lsequence_list.append(sub_raw_bert_indices1)
            bert_N_S_segments_ids = [0] * (bert_N_S_length + 2)
            bert_sub_segments_ids = [0] * (bert_sub_length + 2)
            if (len(bert_sub_segments_ids)!=len(sub_raw_bert_indices1)):

                print(len(bert_sub_segments_ids),len(sub_raw_bert_indices))
            bert_N_S_segments_ids_list.append(bert_N_S_segments_ids)
            bert_sub_segments_ids_list.append(bert_sub_segments_ids)
            asp = list(aspect['aspect_term'])
            asp_bert_ids, _, _ = text2bert_id(asp, tokenizer)
            bert_sequence = CLS_id + text_raw_bert_indices + SEP_id + asp_bert_ids + SEP_id##[cls]句子[sep]方面词[sep]
            bert_segments_ids = [0] * (bert_length + 2) + [1] * (len(asp_bert_ids) + 1)
            aspect_indi = [0] * length
            adj_i_oneshot = head_to_adj_oneshot(dep_head, length, d['aspect_terms'])
            rel_adj = rel_to_adj_oneshot(dep_rel,dep_head,length, d['aspect_terms'],edge_dict)
            forward_adj,backward_adj,forward_adj1,backward_adj1,mask_adj = abcd(dep_rel,dep_head,d['sub_start_end'],rel_dict, edge_dict)
            dep_rel_temp = [edge_dict[dep] for dep in dep_rel]
            rel_adj_bert = rel_adj.flatten().tolist()
            word_mapback_rel = [iiii for iiii in range(len(rel_adj_bert))]
            rel_adj_bert_length = len(word_mapback_rel)
            bert_rel_sequence_indices1 = CLS_id + rel_adj_bert + SEP_id
            bert_rel_segments_ids = [0] * (rel_adj_bert_length + 2)
            word_mapback_rel_list.append(word_mapback_rel)
            rel_adj_bert_length_list.append(rel_adj_bert_length)
            bert_rel_sequence_indices1_list.append(bert_rel_sequence_indices1)
            bert_rel_segments_ids_list.append(bert_rel_segments_ids)

            nonzero_mask = (rel_adj != 0)
            rel_adj_mask_list.append(np.array(nonzero_mask,dtype=np.float32))
            rel_adj_list.append(rel_adj)

            adj.append(adj_i_oneshot)
            for pidx in range(aspect['left_index'], aspect['right_index']):
                aspect_indi[pidx] = 1

            if all(element == 0 for element in aspect_indi):
                print(aspect_indi)
                print(aspect['left_index'], aspect['right_index'])
            aspect_indi_list.append(aspect_indi)
            sub_length_list.append(len(d['sub_start_end']))
            sub_adj_list_temp = []
            for sub_graph_index in d['sub_start_end']:
                sub_graph_adj = np.zeros((length, length), dtype=np.float32)
                sub_graph_adj[sub_graph_index[0]:sub_graph_index[1],sub_graph_index[0]:sub_graph_index[1]] = \
                    adj_i_oneshot[sub_graph_index[0]:sub_graph_index[1],sub_graph_index[0]:sub_graph_index[1]]
                sub_adj_list_temp.append(sub_graph_adj)
            sub_adj_list.append(sub_adj_list_temp)
            bert_sequence = bert_sequence[:max_len + 3]
            bert_segments_ids = bert_segments_ids[:max_len + 3]
            sub_index = d['sub_start_end']
            # 方面词在句子中的位置
            label_list.append(label)
            bert_sequence_list.append(bert_sequence)
            bert_segments_ids_list.append(bert_segments_ids)
            if(word_mapback == []):

                # print("word_mapback")
                raise Exception
            if(word_mapback_rel == []):
                # print("word_mapback_rel")
                raise Exception
            if(word_mapback_sub == []):
                # print("word_mapback_sub")
                raise Exception
            if(word_mapback_N_S == []):
                # print("word_mapback_N_S")
                raise Exception
            mapback_list.append(word_mapback)
            sub_index_list.append(sub_index)

    processed = {
            'length': length_list,####句子的长度
            'bert_length': bert_length_list,##bert的长度
            'word_mapback': mapback_list,###bert的token映射为原word
            'bert_sequence_list': bert_sequence_list,###bert的序列
            'bert_segments_ids_list': bert_segments_ids_list,###bert的分句
            'label_list': label_list,
            'aspect_nodeindex_list': aspect_indi_list,##方面词索引
            'relation_list': relation_list,##（这个是无用项）
            'adj_list':adj,###整个句子邻接矩阵
            'bert_N_S_length_list':bert_N_S_length_list,###N_S的长度 N_S代表nuclear和satellite
            'bert_sub_length_list':bert_sub_length_list,###子图节点长度
            'sub_adj_list':sub_adj_list,###子图邻接矩阵合集（这个是无用项）
            'bert_N_S_sequence_list':bert_N_S_sequence_list,##N_S矩阵转成文本序列 目的是获取nuclear和satellite的初始表征
            'bert_sub_sequence_list':bert_sub_lsequence_list,##sub矩阵转成文本序列 目的是获取子句关系的初始表征
            'sub_rel_adj_list':sub_rel_adj_list,##代表子句关系的邻接矩阵 有向的
            'word_mapback_N_S_list':word_mapback_N_S_list, #与word_mapback的含义类似，也是用来映射回原单词
            'word_mapback_sub_list':word_mapback_sub_list,
            'rel_adj_list': rel_adj_list,##这是句法依赖的关系矩阵
            'bert_N_S_segments_ids_list':bert_N_S_segments_ids_list,
            'bert_sub_segments_ids_list':bert_sub_segments_ids_list,
            'node_num_list':node_num_list, ##这是子句数量
            'sub_graph_ind_list':sub_graph_ind_list, ##哪些单词位于该子句中，用于获取子句表征用的
             'rel_adj_mask_list':rel_adj_mask_list, ## 句法依赖的关系邻接矩阵
             'aspect_in_sub_list':aspect_in_sub_list,###方面词所在子句的位置
             'adj_matrix_pos_list':adj_matrix_pos_list,###每个单词的词性
             'rel_list':rel_list,##依赖边，list版本
        'word_mapback_rel_list': word_mapback_rel_list, ##无用项
            'rel_adj_bert_length_list':rel_adj_bert_length_list,##无用项
        'bert_rel_sequence_indices1_list':bert_rel_sequence_indices1_list,##无用项
           " bert_rel_segments_ids_list": bert_rel_segments_ids_list,##无用项
          "all_aspect":all_aspect, ##方面词所在句子中的位置
           'all_aspect_label':all_aspect_label, ##负例方面词mask
            'aspect_num_list':aspect_num_list, ##方面词数量
            'all_aspect_label_reverse':all_aspect_label_reverse, ##正例方面词mask
            'all_aspect_label_a':all_aspect_label_a, ##无用项
             'all_clause':all_clause,##方面词所在子句的位置
             "all_clause_label":all_clause_label, ##方面词所在的子句负例mask
             "sub_length_list":sub_length_list, ##子句个数
             'all_clause_re_label':all_clause_re_label,##方面词所在的子句正例mask
             'clause_label_super_label_list':clause_label_super_label_list, ##所有子句的负例mask
             'clause_label_super_reverse_label_list':clause_label_super_reverse_label_list ##所有子句的正例mask
    }
    return processed

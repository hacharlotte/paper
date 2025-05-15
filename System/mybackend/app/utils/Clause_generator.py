import os
import torch
import numpy as np
import argparse
import os
from app.utils import config
import json
from transformers import AutoTokenizer, AutoModel
from app.utils.model_depth import ParsingNet
from transformers import BertTokenizer
from supar import Parser
from app.utils.parameter_for_sentiment import get_parameter
from app.models.zhongjires16 import MCDG
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.global_gpu_id)


def parse_args():
    parser = argparse.ArgumentParser()
    """ config the saved checkpoint """
    parser.add_argument('--ModelPath', type=str, default=r'D:\NLP\DMRST_Parser-main\DMRST_Parser-main\depth_mode\Savings\multi_all_checkpoint.torchsave', help='pre-trained model')
    base_path = config.tree_infer_mode + "_mode/"
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--savepath', type=str, default=base_path + './Savings', help='Model save path')
    args = parser.parse_args()
    return args


def inference(model, tokenizer, input_sentences, batch_size):
    LoopNeeded = int(np.ceil(len(input_sentences) / batch_size))

    input_sentences = [tokenizer.tokenize(i, add_special_tokens=False) for i in input_sentences]
    all_segmentation_pred = []
    all_tree_parsing_pred = []

    with torch.no_grad():
        for loop in range(LoopNeeded):
            StartPosition = loop * batch_size
            EndPosition = (loop + 1) * batch_size
            if EndPosition > len(input_sentences):
                EndPosition = len(input_sentences)

            input_sen_batch = input_sentences[StartPosition:EndPosition]
            _, _, SPAN_batch, _, predict_EDU_breaks = model.TestingLoss(input_sen_batch, input_EDU_breaks=None, LabelIndex=None,
                                                                        ParsingIndex=None, GenerateTree=True, use_pred_segmentation=True)
            all_segmentation_pred.extend(predict_EDU_breaks)
            all_tree_parsing_pred.extend(SPAN_batch)
    return input_sentences, all_segmentation_pred, all_tree_parsing_pred
import nltk
def parse_xml(path, lowercase=True, remove_list=None):
    # file_out = open(r"data\text_for_inference.txt", "w", encoding="utf-8")
    dep_parser = Parser.load(r'D:\NLP\BiSyn_GAT_plus\BiSyn_GAT_plus\ptb.biaffine.dep.lstm.char')
    if remove_list is None:
        remove_list = []
    dataset = []
    for sentence in path:
        sent = sentence['text']
        if lowercase:
            sent = sent.lower()
        terms = sentence['events']
        if terms is None:
            continue
        accept_terms = []
        for term in terms:
            aspect = term['v']
            sentiment = "neutral"
            implicit = False
            if sentiment in remove_list:
                continue
            left_index = int(term['from'])
            right_index = int(term['to'])
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
        dict1 = {}
        dict1['aspect_terms'] = accept_terms
        dict1['text'] = sent
        dict1['token'] = sentence['token']
        dict1['sub_start_end'] = sentence["sub_start_end"]
        dict1['sub_rel_list'] = sentence["sub_rel_list"]
        datase = dep_parser.predict(dict1['token'], verbose=False)
        dep_head = datase.arcs[0]
        dep_rel = datase.rels[0]
        dict1['dep_head'] = [x - 1 for x in dep_head]
        dict1['dep_rel'] = dep_rel
        dict1['pos_tag'] = nltk.pos_tag(sentence['token'])
        if accept_terms:
            dataset.append(dict1)
    return dataset
import random
def set_random_seed(args):
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader
from app.utils.ABSA_dataset_v3 import ABSA_Dataset,ABSA_collate_fn
from app.utils.preprocess_for_sentiment import process
from sklearn import metrics
def evaluate(model, dataloader, args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    predictions, labels = [], []

    val_loss, val_acc = 0.0, 0.0
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
        # model.eval()
        # with torch.no_grad():
            length_, bert_length_, word_mapback, adj_oneshot, \
                adj_oneshot1, adj_oneshot2, \
                bert_segments_ids_list, word_mapback_N_S_list, \
                word_mapback_sub_list, bert_sub_sequence_list, \
                bert_N_S_sequence_list, aspect_masks, \
                bert_tokens, adj_oneshot3, max_node_num, bert_N_S_length_list, bert_sub_length_list, bert_N_S_segments_ids_list, bert_sub_segments_ids_list,adj_oneshot4,aspect_in_sub_list,adj_oneshot5,all_aspect_adj,all_aspect_label_adj,all_aspect_label_adj_reverse,all_aspect_label_adj_aa,all_clause_adj,all_clause_label_adj,all_clause_re_label_adj,all_clause_super_label_adj,all_clause_super_re_label_adj,labels_ = batch
            batch = (length_.to('cuda'), bert_length_.to('cuda'), word_mapback.to('cuda'), adj_oneshot.to('cuda'),adj_oneshot1.to('cuda'), adj_oneshot2.to('cuda'),bert_segments_ids_list.to('cuda'), word_mapback_N_S_list.to('cuda'),word_mapback_sub_list.to('cuda'), bert_sub_sequence_list.to('cuda'),bert_N_S_sequence_list.to('cuda'), aspect_masks.to('cuda'),bert_tokens.to('cuda'), adj_oneshot3.to('cuda'), max_node_num.to('cuda'), bert_N_S_length_list.to('cuda'), bert_sub_length_list.to('cuda'), bert_N_S_segments_ids_list.to('cuda'), bert_sub_segments_ids_list.to('cuda'),adj_oneshot4.to('cuda'),aspect_in_sub_list.to('cuda'),adj_oneshot5.to('cuda'),all_aspect_adj.to('cuda'),all_aspect_label_adj.to('cuda'),all_aspect_label_adj_reverse.to('cuda'),all_aspect_label_adj_aa.to('cuda'),all_clause_adj.to('cuda'),all_clause_label_adj.to('cuda'),all_clause_re_label_adj.to('cuda'),all_clause_super_label_adj.to('cuda'),all_clause_super_re_label_adj.to('cuda'),labels_.to('cuda'))
            inputs = batch
            label = labels_.to('cuda')

            logits,_,_ = model(inputs)
            logits = logits.squeeze(1)
            labels1 = label
            bert_tokens = bert_tokens.to('cuda')
            predictions += np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
            labels += label.data.cpu().numpy().tolist()
            # pred = logits.argmax(dim=-1)
            # for index,i in enumerate(bert_tokens):
            #     tokens = tokenizer.convert_ids_to_tokens(i)
            #     # print(tokens)
            #     text = ''
            #     sum = 0
            #     for j in tokens:
            #         if j == '[CLS]':
            #             continue
            #         elif j == '[SEP]':
            #             if sum == 0:
            #                 textprint = text
            #             else:
            #                 aspectprint = text
            #             text = ''
            #             sum = 1
            #         elif j == '[PAD]':
            #             break
            #         else:
            #             text = text + j + ' '
            #     print(textprint,aspectprint,pred[pred!=labels1][index],labels1[pred!=labels1][index])

    return predictions
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*apply_permutation.*")
warnings.filterwarnings("ignore", message=".*attention_mask.*")
def generate(text):
    import re
    from transformers import AutoTokenizer, AutoModel
    from nltk.tokenize import word_tokenize
    import torch

    args = parse_args()
    model_path = args.ModelPath
    batch_size = args.batch_size
    save_path = args.savepath

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(r"D:\NLP\DMRST_Parser-main\DMRST_Parser-main\xlm-roberta-base", use_fast=True)
    bert_model = AutoModel.from_pretrained(r"D:\NLP\DMRST_Parser-main\DMRST_Parser-main\xlm-roberta-base").cuda()
    for param in bert_model.parameters():
        param.requires_grad = False

    model = ParsingNet(bert_model, bert_tokenizer=tokenizer).cuda()
    model.load_state_dict(torch.load(model_path), strict=False)
    model.eval()

    test_sentences = [entry["text"] for entry in text]
    input_sentences, segmentation_preds, tree_parsing_preds = inference(model, tokenizer, test_sentences, batch_size)

    output_json = []
    for idx, sentence in enumerate(test_sentences):
        modified = sentence.replace("\u00a0", " ").rstrip()
        tokens = [tok for tok in modified.split(" ") if tok]
        word_tokens = []
        word_indices = []

        original_words = word_tokenize(sentence)
        for i, word in enumerate(tokens):
            sub_tokens = word_tokenize(word)
            if len(sub_tokens) == 2 and sub_tokens[-1] == "." and i != len(tokens) - 1:
                joined = sub_tokens[0] + sub_tokens[1]
                if joined in original_words:
                    sub_tokens = [joined]
            word_tokens.extend(sub_tokens)
            word_indices.extend([i + 1] * len(sub_tokens))

        # Wordpiece to word index mapping (XLM-R)
        wp_to_word = []
        word_id = 0
        for token in input_sentences[idx]:
            if token.startswith("▁"):
                word_id += 1
            wp_to_word.append(word_id)

        # Segment boundary calculation
        segments = []
        span_indices = []
        prev = -1
        for seg_idx in segmentation_preds[idx]:
            try:
                last = len(word_indices) - 1 - word_indices[::-1].index(wp_to_word[seg_idx])
                segments.append(word_indices[prev + 1:last + 1])
                span_indices.append([prev + 1, last + 1])
                prev = last
            except ValueError:
                continue

        # Parse tree relations
        tree_info = []
        if tree_parsing_preds[idx][0] != "NONE":
            for relation in tree_parsing_preds[idx][0].split():
                matches = re.findall(r"(\d+):([^=]+)=([^:]+):(\d+)", relation)
                if matches:
                    tuples = [(int(m[0]), int(m[3]), m[1], m[2]) for m in matches]
                    tree_info.append(tuples)

        output_json.append({
            "text": sentence,
            "sub_start_end": span_indices,
            "token": word_tokens,
            "sub_rel_list": tree_info,
            "events": text[idx]["events"]
        })

    dataset = parse_xml(output_json)
    edge_dict = {}
    pos_dict = {}
    sum = 1
    sum1 = 0
    for i in dataset:
        for j in i['dep_rel']:
            if j not in edge_dict:
                edge_dict[j] = sum
                sum += 1
        for j in i['pos_tag']:
            if j[1] not in pos_dict:
                pos_dict[j[1]] = sum1
                sum1 += 1
    tokenizer = BertTokenizer.from_pretrained(r'D:\NLP\RGAT-ABSA-master\RGAT-ABSA-master\bert-base-uncased')
    processed = process(dataset, tokenizer, edge_dict, pos_dict)
    args = get_parameter()
    set_random_seed(args)
    args.A = 0 #loss
    args.B = 0 #loss
    test_loader = DataLoader(ABSA_Dataset(processed),
                             batch_size=len(processed),
                             shuffle=False,
                             num_workers=0,
                             collate_fn=ABSA_collate_fn)
    model = MCDG(args).to(device=args.device)
    model.load_state_dict(
        torch.load(r'D:\NLP\RGAT-ABSA-master\RGAT-ABSA-master\result\laptop\epoch_1_acc_83.22884013_f1_0.80569697_loss_0.48146486.pt'),
        strict=False
    )
    predict = evaluate(model, test_loader, args)
    return predict, output_json
# if __name__ == '__main__':
#
#     args = parse_args()
#     model_path = args.ModelPath
#     batch_size = args.batch_size
#     save_path = args.savepath
#
#     """ BERT tokenizer and model """
#     bert_tokenizer = AutoTokenizer.from_pretrained(r"D:\NLP\DMRST_Parser-main\DMRST_Parser-main\xlm-roberta-base", use_fast=True)
#     bert_model = AutoModel.from_pretrained(r"D:\NLP\DMRST_Parser-main\DMRST_Parser-main\xlm-roberta-base")
#
#     bert_model = bert_model.cuda()
#
#     for name, param in bert_model.named_parameters():
#         param.requires_grad = False
#
#     model = ParsingNet(bert_model, bert_tokenizer=bert_tokenizer)
#
#     model = model.cuda()
#     model.load_state_dict(torch.load(model_path), strict=False)
#     model = model.eval()
#
#     # 打印每个token对应的原始单词
#     # print(word_mapping)
#
#     Test_InputSentences = []
#     output_json = []
#     input_sentences, all_segmentation_pred, all_tree_parsing_pred = inference(model, bert_tokenizer, Test_InputSentences, batch_size)
#
#     max1 = 1
#     word_mapping_list = []
#     word = []
#     filtered_list_1 = []
#     word1 = []
#     text1 = []
#
#     for i in Test_InputSentences:
#         original_words = word_tokenize(i)
#         original_string = i
#         modified_string = original_string.replace("\u00a0", " ")
#         cc = modified_string[:-1].split(' ')
#         filtered_list = [item for item in cc if item != '']
#         filtered_list_1.append(filtered_list)
#         word.append(cc)
#         word_mapping1 = []
#         word_mapping2 = []
#         text1.append(i)
#         for index, j in enumerate(filtered_list):
#             ccc = word_tokenize(j)
#             if len(ccc)==2 and ccc[-1]=='.' and index!=len(filtered_list)-1:
#                 print(ccc)
#                 temp = []
#                 temp.append(ccc[0] + ccc[1])
#                 if temp[0] in original_words:
#                     ccc = temp
#             word_mapping2 += ccc
#             word_mapping1 += [index + 1] * len(ccc)
#         word1.append(word_mapping2)
#         word_mapping_list.append(word_mapping1)
#     for index,i in enumerate(all_tree_parsing_pred):
#         cccc = []
#         aaaa = []
#         bbbb = []
#         pre = -1
#         j1 = 0
#         word_mapping = []
#         for i1, token in enumerate(input_sentences[index]):
#             original_word = j1 = j1 + 1 if token.startswith('▁') else j1
#             word_mapping.append(original_word)
#         for i1 in all_segmentation_pred[index]:
#             last_index = len(word_mapping_list[index]) - word_mapping_list[index][::-1].index(word_mapping[i1]) - 1
#             aaaa.append(word_mapping_list[index][pre+1:last_index+1])
#             bbbb.append([pre+1,last_index+1])
#             pre = last_index
#         for j in i[0].split(' '):
#             import re
#
#             # 输入的字符串
#             input_str = j
#
#             # 使用正则表达式提取数字、文本和数字
#             matches = re.findall(r'(\d+):([^=]+)=([^:]+):(\d+)', input_str)
#
#             # 将匹配结果转换为元组
#             result_tuples = [(int(match[0]), int(match[3]), match[1], match[2]) for match in matches]
#             cccc.append(result_tuples)
#
#         for index1,jk in enumerate(aaaa):
#             if len(jk)==0:
#                 continue
#             print(word1[index][bbbb[index1][0]:bbbb[index1][1]])
#             #print(word1[index][jk[0]-1:jk[-1]])
#         if i[0] != 'NONE':
#             a = int(i[0].split(' ')[0][-2])
#             max1 = a if a > max1 else max1
#         dict_out = {}
#         dict_out['sub_start_end'] = bbbb
#         dict_out['token'] = word1[index]
#         dict_out['sub_rel_list'] = cccc
#         output_json.append(dict_out)
#     f = open('mams_test.json','w',encoding='utf-8')
#     json.dump(output_json,f)
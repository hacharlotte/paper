import json
import spacy
import nltk
from spacy.tokens import Doc
import json
import pickle
from lxml import etree
from transformers import BertTokenizer
from tqdm import tqdm
from supar import Parser
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
nlp = spacy.load("en_core_web_sm")
# sentiment_dict = {
#     'positive': 0,
#     'negative': 1,
#     'neutral': 2,
# }
sentiment_dict1 = {
    '1': 'positive',
    '-1': 'negative',
    '0': 'neutral',
}

def deal_twitter(raw_file):
    json_du = open(raw_file, 'r', encoding='utf-8')
    json_du = json_du.readlines()
    out = open('res15_train_temp.json','w',encoding='utf-8')

    new = []
    j = 0
    while j < len(json_du):
        new_dict = {}
        sentence = json_du[j][:-1]
        left = sentence.find('$T$')
        j += 1
        aspect = json_du[j][:-1]
        right = len(aspect) + left
        j += 1
        sentiment = json_du[j][:-1]
        j += 1
        sentence = sentence.replace('$T$',aspect)
        token1 = word_tokenize(sentence)
        new_dict['text'] = sentence
        new_dict['aspect_terms'] = [{"aspect_term": aspect, "sentiment": sentiment, "left_index": left, "right_index": right}]
        new_dict['token'] = token1
        new.append(new_dict)
    json.dump(new,out)
        # file_out.write(sentence + "\n")
        # doc = nlp(sentence)
        # # 访问句法依存树的根节点
        # output = []
        # dep_head = []
        # dep_rel = []
        # token2 = []
        # doc = Doc(nlp.vocab, words=token1)
        # # Tagger(doc)
        # for name, tool in nlp.pipeline:
        #     tool(doc)
        # for token in doc:
        #     word = token.text
        #     rel = token.dep_
        #     head_index = token.head.i
        #     # 将结果添加到输出列表
        #     output.append([word, rel, head_index])
        #     dep_head.append(head_index)
        #     dep_rel.append(rel)
        #     token2.append(word)
        # if token1 != token2:
        #     print("1",token1)
        #     print("2",token2)
# deal_twitter(r'D:\NLP\new_RG\RGAT-ABSA-master\data\twitter\test.raw')
# deal_twitter(r'rest15_train.raw')
def ha():
    b1 = open('../res15_test_new1.json', 'w')
    a = open('res15_test_new.json', 'r')
    a = json.load(a)
    b = []
    pre = '0'
    for i in a:
        sent = i['text']
        se = i['aspect_terms'][0]['sentiment']
        i['aspect_terms'][0]['sentiment'] = sentiment_dict1[se]
        left_index = i['aspect_terms'][0]['left_index']
        right_index = i['aspect_terms'][0]['right_index']
        left_word_offset = len(word_tokenize(sent[:left_index]))
        to_word_offset = len(word_tokenize(sent[:right_index]))
        if left_word_offset == to_word_offset:
            left_word_offset = left_word_offset - 1
        i['aspect_terms'][0]['left_index'] = left_word_offset
        i['aspect_terms'][0]['right_index'] = to_word_offset
        if i['text'] == pre:
            b[-1]['aspect_terms'].append(i['aspect_terms'][0])
            print(i['text'])
        else:
            b.append(i)
            pre = i['text']
    json.dump(b, b1)
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
def parse_xml(path, lowercase=True, remove_list=None):
    # file_out = open(r"data\text_for_inference.txt", "w", encoding="utf-8")
    if remove_list is None:
        remove_list = []
    dataset = []
    with open(path, 'rb') as f:
        root = etree.fromstring(f.read())
        for sentence in root:
            # index = sentence.get('id')
            sent = sentence.find('text').text
            if lowercase:
                sent = sent.lower()
            # file_out.write(sent + "\n")
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
                    'text': sent,
                    'aspect_terms': accept_terms,
                })
    return dataset
def pre_data():
    path = r'train.xml'
    path2 = r'mams_train.json'
    json_du = open('mams_train_new.json', 'w', encoding='utf-8')
    dataset1 = parse_xml(path, lowercase=True)
    dataset2 = parse_json(path2)
    dataset = dataset_end(dataset1, dataset2)
    json.dump(dataset, json_du)
pre_data() ##得到带句法依赖的数据
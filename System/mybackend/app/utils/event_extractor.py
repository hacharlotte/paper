# app/utils/event_extractor.py

from transformers import BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


def build_event_features_from_llm(events_from_llm):
    """
    输入: LLM 返回的事件结构，格式如下：
    [
        {
            "sentence": ["Barack", "Obama", "visited", "Japan", "."],
            "event": {"s": "Barack Obama", "v": "visited", "o": "Japan", "p": null},
            "trigger_index": 2
        },
        ...
    ]

    返回:
        - sentences: 所有句子的word级别分词
        - enc_input_ids: 每个句子的token ids（含[CLS]和[SEP]）
        - enc_mask_ids: 每个句子的mask
        - node_event: 每个事件在句子中的token span位置
    """
    sentences = []  # word list
    enc_input_ids = []  # tensor[]
    enc_mask_ids = []  # tensor[]
    node_event = []

    for item in events_from_llm:
        sentence = item["sentence"]
        trigger_index = item["trigger_index"]

        # 保存原始 word sentence
        if sentence not in sentences:
            sentences.append(sentence)

        sent_id = sentences.index(sentence)

        # word to token mapping
        word_tokens = [tokenizer.tokenize(w) for w in sentence]
        word_token_lens = [len(wt) for wt in word_tokens]
        tokens = [token for sublist in word_tokens for token in sublist]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # 构造 input_ids, mask
        enc_ids = torch.tensor([tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id])
        mask_ids = torch.tensor([1] * len(enc_ids))

        # 对应句子只加一次
        if sent_id >= len(enc_input_ids):
            enc_input_ids.append(enc_ids)
            enc_mask_ids.append(mask_ids)

        # trigger span
        start_token = sum(word_token_lens[:trigger_index])
        end_token = start_token + word_token_lens[trigger_index]

        node_event.append([sent_id, [start_token, end_token]])

    return sentences, enc_input_ids, enc_mask_ids, node_event

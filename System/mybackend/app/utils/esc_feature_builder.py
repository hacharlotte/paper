# app/utils/esc_feature_builder.py

import requests
from transformers import BertTokenizer
import torch
# from app.utils.event_extractor import build_event_features_from_llm
from app.utils.esc_features import ESC_features
from openai import OpenAI
# 建议放到环境变量里
DEEPSEEK_API_KEY = "sk-ad50785559304e64ab8457892fb06174"

tokenizer = BertTokenizer.from_pretrained(r"D:\NLP\iLIF-master\iLIF-master\bert-base-uncased")

def build_event_features_from_llm(sentences, events,text, max_len=120, topic_id="test", doc_id="test"):
    node_event = []
    enc_input_ids = []
    enc_mask_ids = []

    for sentence in sentences:
        sent_str = ' '.join(sentence)  # 还原成句子
        encode_dict = tokenizer.encode_plus(
            sent_str,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_len,
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        enc_input_ids.append(encode_dict['input_ids'].squeeze(0))   # [seq_len]
        enc_mask_ids.append(encode_dict['attention_mask'].squeeze(0))
    # stack 成 batch tensor
    enc_input_ids = torch.stack(enc_input_ids)   # shape: [batch_size, seq_len]
    enc_mask_ids = torch.stack(enc_mask_ids)     # shape: [batch_size, seq_len]
    # 构建 node_event
    for event in events:
        sent_id = event['sentence_id']
        trigger_index = event['trigger_index']

        # word2token
        word_tokens = [tokenizer.tokenize(w) for w in sentences[sent_id]]
        word_token_lens = [len(wt) for wt in word_tokens]

        # word_index --> token span
        start_token = sum(word_token_lens[:trigger_index]) + 1   # +1 是因为 encode_plus 多了 [CLS]
        end_token = start_token + word_token_lens[trigger_index]

        node_event.append([sent_id, [start_token, end_token]])
    # Step3 - 事件对
    event_pairs = [[i, j] for i in range(len(node_event)) for j in range(i+1, len(node_event))]

    # Step4 - ESC_features
    features = ESC_features(
        topic_id=topic_id,
        doc_id=doc_id,
        enc_text=text,
        enc_tokens=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)),
        sentences=sentences,
        enc_input_ids=enc_input_ids,
        enc_mask_ids=enc_mask_ids,
        node_event=node_event,
        t1_pos=[i for i, j in event_pairs],
        t2_pos=[j for i, j in event_pairs],
        target=[0]*len(event_pairs),
        rel_type=[0]*len(event_pairs),
        event_pairs=event_pairs
    )
    return features


client = OpenAI(
    api_key=DEEPSEEK_API_KEY,  # 请替换成你的 key
    base_url="https://api.deepseek.com"
)

def call_deepseek_event_extractor(text):
    prompt = f"""
    请执行以下两步：

    1. 请将下列文本分句，输出句子的分词结果，格式如下：
    "sentences": [["词1", "词2", ...], ["词1", "词2", ...], ...]

    2. 请抽取文本中的所有事件，每个事件返回：
    {{
       "event": {{"s": ..., "v": ..., "o": ..., "p": ...}},
       "sentence_id": 句子编号（从0开始），
       "trigger_index": 事件的触发词 v 在该句子中的词索引（从0开始）
    }}

    最终以 JSON 格式输出：
    {{
       "sentences": [...],
       "events": [...]
    }}
    其中v是描述事件的谓词动词，s和o分别是其主语和宾语, p是介词宾语。
    文本：{text}
    """

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )

    content = response.choices[0].message.content.strip()
    # ----- 重点：移除 markdown code block -----
    if content.startswith("```"):
        content = content.strip("```json").strip("```").strip()
    import json
    output = json.loads(content)
    sentences = output['sentences']
    events_from_llm = output['events']

    return sentences, events_from_llm


def build_ESC_features(text, topic_id="test", doc_id="test"):
    """
    一键生成 ESC_features
    """
    # Step1 - 事件抽取
    sentences, events_from_llm = call_deepseek_event_extractor(text)

    # Step2 - 构建 feature 所需字段
    sentences, enc_input_ids, enc_mask_ids, node_event = build_event_features_from_llm(sentences, events_from_llm)

    # Step3 - 事件对
    event_pairs = [[i, j] for i in range(len(node_event)) for j in range(i+1, len(node_event))]

    # Step4 - ESC_features
    features = ESC_features(
        topic_id=topic_id,
        doc_id=doc_id,
        enc_text=text,
        enc_tokens=tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text)),
        sentences=sentences,
        enc_input_ids=enc_input_ids,
        enc_mask_ids=enc_mask_ids,
        node_event=node_event,
        t1_pos=[i for i, j in event_pairs],
        t2_pos=[j for i, j in event_pairs],
        target=[0]*len(event_pairs),
        rel_type=[0]*len(event_pairs),
        event_pairs=event_pairs
    )

    return features, {"sentences": sentences, "events": events_from_llm}
# if __name__ == '__main__':
#     output = {
#   "sentences": [
#     ["Lindsay", "Lohan", "Leaves", "Betty", "Ford", ",", "Checks", "Into", "Malibu", "Rehab"],
#     ["First", "Published", ":", "June", "13", ",", "2013", "4", ":", "59", "PM", "EDT"],
#     ["Lindsay", "Lohan", "has", "left", "the", "Betty", "Ford", "Center", "and", "is", "moving", "to", "a", "rehab", "facility", "in", "Malibu", ",", "Calif", ".", ",", "Access", "Hollywood", "has", "confirmed", "."],
#     ["A", "spokesperson", "for", "The", "Los", "Angeles", "Superior", "Court", "confirmed", "to", "Access", "that", "a", "judge", "signed", "an", "order", "yesterday", "allowing", "the", "transfer", "to", "Cliffside", ",", "where", "she", "will", "continue", "with", "her", "90", "-", "day", "court", "-", "mandated", "rehab", "."],
#     ["Lohan", "'", "s", "attorney", ",", "Shawn", "Holley", ",", "spoke", "out", "about", "the", "move", "."],
#     ["\"", "Lindsay", "is", "grateful", "for", "the", "treatment", "she", "received", "at", "the", "Betty", "Ford", "Center", "."],
#     ["She", "has", "completed", "her", "course", "of", "treatment", "there", "and", "looks", "forward", "to", "continuing", "her", "treatment", "and", "building", "on", "the", "foundation", "established", "at", "Betty", "Ford", ",", "\"", "Holley", "said", "in", "a", "statement", "to", "Access", "."],
#     ["The", "actress", "checked", "into", "the", "Betty", "Ford", "Center", "in", "May", "as", "part", "of", "a", "plea", "deal", "stemming", "from", "her", "June", "2012", "car", "accident", "case", "."]
#   ],
#   "events": [
#     {
#       "event": {
#         "s": "Lindsay Lohan",
#         "v": "Leaves",
#         "o": "Betty Ford",
#         "p": ""
#       },
#       "sentence_id": 0,
#       "trigger_index": 2
#     },
#     {
#       "event": {
#         "s": "Lindsay Lohan",
#         "v": "Checks",
#         "o": "Malibu Rehab",
#         "p": ""
#       },
#       "sentence_id": 0,
#       "trigger_index": 6
#     },
#     {
#       "event": {
#         "s": "Lindsay Lohan",
#         "v": "left",
#         "o": "the Betty Ford Center",
#         "p": ""
#       },
#       "sentence_id": 2,
#       "trigger_index": 3
#     },
#     {
#       "event": {
#         "s": "Lindsay Lohan",
#         "v": "moving",
#         "o": "a rehab facility",
#         "p": "Malibu , Calif"
#       },
#       "sentence_id": 2,
#       "trigger_index": 10
#     },
#     {
#       "event": {
#         "s": "Access Hollywood",
#         "v": "confirmed",
#         "o": "",
#         "p": ""
#       },
#       "sentence_id": 2,
#       "trigger_index": 24
#     },
#     {
#       "event": {
#         "s": "A spokesperson for The Los Angeles Superior Court",
#         "v": "confirmed",
#         "o": "Access",
#         "p": ""
#       },
#       "sentence_id": 3,
#       "trigger_index": 8
#     },
#     {
#       "event": {
#         "s": "a judge",
#         "v": "signed",
#         "o": "an order",
#         "p": ""
#       },
#       "sentence_id": 3,
#       "trigger_index": 13
#     },
#     {
#       "event": {
#         "s": "she",
#         "v": "continue",
#         "o": "her 90-day court-mandated rehab",
#         "p": ""
#       },
#       "sentence_id": 3,
#       "trigger_index": 27
#     },
#     {
#       "event": {
#         "s": "Lohan's attorney, Shawn Holley",
#         "v": "spoke",
#         "o": "",
#         "p": "the move"
#       },
#       "sentence_id": 4,
#       "trigger_index": 8
#     },
#     {
#       "event": {
#         "s": "Lindsay",
#         "v": "is",
#         "o": "grateful",
#         "p": "the treatment"
#       },
#       "sentence_id": 5,
#       "trigger_index": 2
#     },
#     {
#       "event": {
#         "s": "she",
#         "v": "received",
#         "o": "treatment",
#         "p": "the Betty Ford Center"
#       },
#       "sentence_id": 5,
#       "trigger_index": 8
#     },
#     {
#       "event": {
#         "s": "She",
#         "v": "completed",
#         "o": "her course of treatment",
#         "p": ""
#       },
#       "sentence_id": 6,
#       "trigger_index": 2
#     },
#     {
#       "event": {
#         "s": "She",
#         "v": "looks",
#         "o": "",
#         "p": "continuing her treatment and building on the foundation established at Betty Ford"
#       },
#       "sentence_id": 6,
#       "trigger_index": 5
#     },
#     {
#       "event": {
#         "s": "Holley",
#         "v": "said",
#         "o": "",
#         "p": "a statement to Access"
#       },
#       "sentence_id": 6,
#       "trigger_index": 27
#     },
#     {
#       "event": {
#         "s": "The actress",
#         "v": "checked",
#         "o": "the Betty Ford Center",
#         "p": "May"
#       },
#       "sentence_id": 7,
#       "trigger_index": 2
#     }
#   ]
# }
#     sentences = output['sentences']  # ✅ 完整的句子
#     events_from_llm = output['events']  # ✅ 事件列表
#     build_event_features_from_llm(sentences, events_from_llm)
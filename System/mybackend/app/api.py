# app/api.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from app.utils import event_utils, tokenizer, esc_features
from app.models import causality_model
from app.utils.esc_feature_builder import build_event_features_from_llm
from app.utils.causal_formatter import CausalFormatter
from app.utils.Clause_generator import generate
from app.utils.ttttt import attach_sentiment_to_filtered_events,sync_sentiment_back_to_events,RAG
from app.utils.infer import predict
from neo4j import GraphDatabase
from pydantic import BaseModel
import json
app = FastAPI()
event_cache = {}
from app.utils.esc_feature_builder import call_deepseek_event_extractor
# CORS 允许前端跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Neo4j 配置
uri = "bolt://localhost:7687"
user = "neo4j"
password = ""
driver = GraphDatabase.driver(uri, auth=(user, password))
@app.get("/entity-graph")
async def entity_graph(name: str):
    query = """
    MATCH (n)
    WHERE toLower(n.label) CONTAINS toLower($name)
    WITH collect(n) AS target_nodes
    UNWIND target_nodes AS n
    OPTIONAL MATCH (n)-[r1]-(event:Event)
    OPTIONAL MATCH (n)-[r2]-(sent:Sentiment)
    RETURN DISTINCT n, r1, event, r2, sent
    LIMIT 1000
    """
    nodes = {}
    edges = []

    with driver.session() as session:
        result = session.run(query, name=name)
        for record in result:
            # ✅ 收集所有节点
            for key in ["n", "event", "sent"]:
                node = record.get(key)
                if node and node.element_id not in nodes:
                    nodes[node.element_id] = {
                        "id": node.element_id,
                        "label": node.get("label") or node.get("name") or "Unnamed",
                        "title": node.get("label") or node.get("name") or "Unnamed",
                        "group": list(node.labels)[0] if node.labels else "Node"
                    }

            # ✅ 收集所有边，注意 element_id
            for rel_key in ["r1", "r2"]:
                rel = record.get(rel_key)
                if rel is not None:
                    try:
                        start_node, end_node = rel.nodes
                        edges.append({
                            "from": start_node.element_id,
                            "to": end_node.element_id,
                            "label": rel.type
                        })
                    except Exception as e:
                        print("❌ 边关系解析失败:", rel, e)

    return {
        "nodes": list(nodes.values()),
        "edges": edges
    }







# API: 返回节点 + 边
@app.get("/graph")
async def get_graph():
    query = """
    MATCH (e:Entity)-[:PARTICIPATE_IN]->(ev:Event)-[:HAS_SENTIMENT]->(s:Sentiment)
    OPTIONAL MATCH (ev)-[:CAUSES]->(ev2:Event)
    RETURN e, ev, s, ev2
    LIMIT 20000
    """

    nodes = {}
    edges = []

    with driver.session() as session:
        results = session.run(query)
        for record in results:
            e = record["e"]
            ev = record["ev"]
            s = record["s"]
            ev2 = record.get("ev2")  # 可能为 None

            for node in [e, ev, s, ev2]:
                if node and node.id not in nodes:
                    nodes[node.id] = {
                        "id": node.id,
                        "label": list(node.labels)[0] if node.labels else "Node",
                        "title": node.get("label", "Unnamed")
                    }

            # 构造关系边
            edges.append({
                "from": e.id,
                "to": ev.id,
                "label": "PARTICIPATE_IN"
            })
            edges.append({
                "from": ev.id,
                "to": s.id,
                "label": "HAS_SENTIMENT"
            })
            if ev2:
                edges.append({
                    "from": ev.id,
                    "to": ev2.id,
                    "label": "CAUSES"
                })

    return {
        "nodes": list(nodes.values()),
        "edges": edges
    }# ====== 前端请求体 ======
class TextInput(BaseModel):
    input_text: str
def format_prediction_output(features, predicted, events):
    if events is None:
        raise ValueError("features.events 不存在，你需要确保 features 有事件列表")

    causal_pairs = []

    for idx, label in enumerate(predicted):
        if label == 1:
            i, j = features.event_pairs[idx]
            e1 = events[i]["event"]  # 只取事件内容，不要 sentence_id
            e2 = events[j]["event"]
            causal_pairs.append({
                "i": i,
                "j": j,
                "e1": e1,
                "e2": e2
            })

    return causal_pairs


# ====== API ======
@app.post("/predict/causality")
def predict_causality(data: TextInput, page: int = 1, page_size: int = 20):
    text = data.input_text.strip()
    if text not in event_cache:
        return {"error": "请先进行事件抽取"}

    sentences = event_cache[text]["sentences"]
    events = event_cache[text]["events"]

    features = build_event_features_from_llm(sentences, events,text)

    prediction = causality_model.predict(features)
    causal_pairs = CausalFormatter.format_pairs(features, prediction, events)
    paged = CausalFormatter.paginate(causal_pairs, page=page, page_size=page_size)
    # causal_pairs = [{'id': 0, 'i': 0, 'j': 1, 'e1': {'s': 'Lindsay Lohan', 'v': 'Leaves', 'o': 'Betty Ford', 'p': None}, 'e2': {'s': 'Lindsay Lohan', 'v': 'Checks', 'o': 'Malibu Rehab', 'p': None}}, {'id': 2, 'i': 0, 'j': 3, 'e1': {'s': 'Lindsay Lohan', 'v': 'Leaves', 'o': 'Betty Ford', 'p': None}, 'e2': {'s': 'Lindsay Lohan', 'v': 'moving', 'o': 'a rehab facility', 'p': 'Malibu'}}, {'id': 25, 'i': 2, 'j': 3, 'e1': {'s': 'Lindsay Lohan', 'v': 'left', 'o': 'the Betty Ford Center', 'p': None}, 'e2': {'s': 'Lindsay Lohan', 'v': 'moving', 'o': 'a rehab facility', 'p': 'Malibu'}}]
    event_cache[text]["causal_pairs"] = causal_pairs
    # events = [{'event': {'s': 'Lindsay Lohan', 'v': 'Leaves', 'o': 'Betty Ford', 'p': None}, 'sentence_id': 0,
    #            'trigger_index': 2},
    #           {'event': {'s': 'Lindsay Lohan', 'v': 'Checks', 'o': 'Malibu Rehab', 'p': None}, 'sentence_id': 0,
    #            'trigger_index': 6},
    #           {'event': {'s': 'Lindsay Lohan', 'v': 'left', 'o': 'the Betty Ford Center', 'p': None}, 'sentence_id': 2,
    #            'trigger_index': 3},
    #           {'event': {'s': 'Lindsay Lohan', 'v': 'moving', 'o': 'a rehab facility', 'p': 'Malibu'}, 'sentence_id': 2,
    #            'trigger_index': 10},
    #           {'event': {'s': 'Access Hollywood', 'v': 'confirmed', 'o': None, 'p': None}, 'sentence_id': 2,
    #            'trigger_index': 24}, {
    #               'event': {'s': 'A spokesperson for The Los Angeles Superior Court', 'v': 'confirmed', 'o': None,
    #                         'p': 'Access'}, 'sentence_id': 3, 'trigger_index': 8},
    #           {'event': {'s': 'a judge', 'v': 'signed', 'o': 'an order', 'p': None}, 'sentence_id': 3,
    #            'trigger_index': 14},
    #           {'event': {'s': 'Shawn Holley', 'v': 'spoke', 'o': None, 'p': 'the move'}, 'sentence_id': 4,
    #            'trigger_index': 8},
    #           {'event': {'s': 'Lindsay', 'v': 'received', 'o': 'the treatment', 'p': 'the Betty Ford Center'},
    #            'sentence_id': 5, 'trigger_index': 8},
    #           {'event': {'s': 'She', 'v': 'completed', 'o': 'her course of treatment', 'p': None}, 'sentence_id': 6,
    #            'trigger_index': 2},
    #           {'event': {'s': 'She', 'v': 'looks', 'o': None, 'p': 'continuing her treatment'}, 'sentence_id': 6,
    #            'trigger_index': 9},
    #           {'event': {'s': 'She', 'v': 'building', 'o': None, 'p': 'the foundation'}, 'sentence_id': 6,
    #            'trigger_index': 15},
    #           {'event': {'s': 'Holley', 'v': 'said', 'o': None, 'p': 'a statement'}, 'sentence_id': 6,
    #            'trigger_index': 26},
    #           {'event': {'s': 'The actress', 'v': 'checked', 'o': 'the Betty Ford Center', 'p': None}, 'sentence_id': 7,
    #            'trigger_index': 2}]
    # sentences = [['Lindsay', 'Lohan', 'Leaves', 'Betty', 'Ford', ',', 'Checks', 'Into', 'Malibu', 'Rehab'],
    #              ['First', 'Published', ':', 'June', '13', ',', '2013', '4', ':', '59', 'PM', 'EDT'],
    #              ['Lindsay', 'Lohan', 'has', 'left', 'the', 'Betty', 'Ford', 'Center', 'and', 'is', 'moving', 'to', 'a',
    #               'rehab', 'facility', 'in', 'Malibu', ',', 'Calif', '.', ',', 'Access', 'Hollywood', 'has',
    #               'confirmed', '.'],
    #              ['A', 'spokesperson', 'for', 'The', 'Los', 'Angeles', 'Superior', 'Court', 'confirmed', 'to', 'Access',
    #               'that', 'a', 'judge', 'signed', 'an', 'order', 'yesterday', 'allowing', 'the', 'transfer', 'to',
    #               'Cliffside', ',', 'where', 'she', 'will', 'continue', 'with', 'her', '90', '-', 'day', 'court', '-',
    #               'mandated', 'rehab', '.'],
    #              ['Lohan', '’', 's', 'attorney', ',', 'Shawn', 'Holley', ',', 'spoke', 'out', 'about', 'the', 'move',
    #               '.'],
    #              ['“', 'Lindsay', 'is', 'grateful', 'for', 'the', 'treatment', 'she', 'received', 'at', 'the', 'Betty',
    #               'Ford', 'Center', '.'],
    #              ['She', 'has', 'completed', 'her', 'course', 'of', 'treatment', 'there', 'and', 'looks', 'forward',
    #               'to', 'continuing', 'her', 'treatment', 'and', 'building', 'on', 'the', 'foundation', 'established',
    #               'at', 'Betty', 'Ford', ',', '”', 'Holley', 'said', 'in', 'a', 'statement', 'to', 'Access', '.'],
    #              ['The', 'actress', 'checked', 'into', 'the', 'Betty', 'Ford', 'Center', 'in', 'May', 'as', 'part',
    #               'of', 'a', 'plea', 'deal', 'stemming', 'from', 'her', 'June', '2012', 'car', 'accident', 'case', '.']]
    # paged = {'total': 3, 'page': 1, 'page_size': 20, 'data': [
    #     {'id': 0, 'i': 0, 'j': 1, 'e1': {'s': 'Lindsay Lohan', 'v': 'Leaves', 'o': 'Betty Ford', 'p': None},
    #      'e2': {'s': 'Lindsay Lohan', 'v': 'Checks', 'o': 'Malibu Rehab', 'p': None}},
    #     {'id': 2, 'i': 0, 'j': 3, 'e1': {'s': 'Lindsay Lohan', 'v': 'Leaves', 'o': 'Betty Ford', 'p': None},
    #      'e2': {'s': 'Lindsay Lohan', 'v': 'moving', 'o': 'a rehab facility', 'p': 'Malibu'}},
    #     {'id': 25, 'i': 2, 'j': 3, 'e1': {'s': 'Lindsay Lohan', 'v': 'left', 'o': 'the Betty Ford Center', 'p': None},
    #      'e2': {'s': 'Lindsay Lohan', 'v': 'moving', 'o': 'a rehab facility', 'p': 'Malibu'}}]}
    return {
        "events": events,
        "sentences": sentences,
        "causal_pairs": paged
    }

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*apply_permutation.*")
warnings.filterwarnings("ignore", message=".*attention_mask.*")

@app.post("/extract/events")
def extract_events(data: TextInput):
    text = data.input_text.strip()
    sentences, events = call_deepseek_event_extractor(text)
    # sentences = [['On', 'April', '11', ',', '2025', ',', 'China', 'announced', 'it', 'would', 'raise', 'tariffs', 'on', 'U.S.', 'imports', 'to', '125%', ',', 'in', 'direct', 'retaliation', 'to', 'U.S.', 'President', 'Donald', "Trump's", 'earlier', 'move', 'to', 'increase', 'tariffs', 'on', 'Chinese', 'goods', 'to', '145%', '.'], ['This', 'escalation', 'marks', 'a', 'significant', 'intensification', 'in', 'the', 'ongoing', 'trade', 'war', 'between', 'the', "world's", 'two', 'largest', 'economies', ',', 'with', 'potential', 'consequences', 'for', 'global', 'supply', 'chains', '.'], ['The', 'White', 'House', 'had', 'continued', 'to', 'pressure', 'China', 'specifically', ',', 'while', 'pausing', 'similar', 'tariff', 'actions', 'against', 'other', 'countries', '.'], ['In', 'response', ',', "China's", 'Finance', 'Ministry', 'condemned', 'the', 'U.S.', 'action', 'as', 'a', 'violation', 'of', 'international', 'trade', 'rules', ',', 'economic', 'principles', ',', 'and', 'described', 'it', 'as', 'unilateral', 'bullying', 'and', 'coercion', '.']]
    # events = [{'event': {'s': 'China', 'v': 'announced', 'o': 'it would raise tariffs on U.S. imports to 125%', 'p': "in direct retaliation to U.S. President Donald Trump's earlier move to increase tariffs on Chinese goods to 145%"}, 'sentence_id': 0, 'trigger_index': 7}, {'event': {'s': 'China', 'v': 'raise', 'o': 'tariffs', 'p': 'on U.S. imports to 125%'}, 'sentence_id': 0, 'trigger_index': 10}, {'event': {'s': 'U.S. President Donald Trump', 'v': 'increase', 'o': 'tariffs', 'p': 'on Chinese goods to 145%'}, 'sentence_id': 0, 'trigger_index': 29}, {'event': {'s': 'This escalation', 'v': 'marks', 'o': "a significant intensification in the ongoing trade war between the world's two largest economies", 'p': 'with potential consequences for global supply chains'}, 'sentence_id': 1, 'trigger_index': 2}, {'event': {'s': 'The White House', 'v': 'continued', 'o': 'to pressure China specifically', 'p': 'while pausing similar tariff actions against other countries'}, 'sentence_id': 2, 'trigger_index': 4}, {'event': {'s': 'The White House', 'v': 'pressure', 'o': 'China', 'p': 'specifically'}, 'sentence_id': 2, 'trigger_index': 7}, {'event': {'s': "China's Finance Ministry", 'v': 'condemned', 'o': 'the U.S. action', 'p': 'as a violation of international trade rules, economic principles'}, 'sentence_id': 3, 'trigger_index': 6}, {'event': {'s': "China's Finance Ministry", 'v': 'described', 'o': 'it', 'p': 'as unilateral bullying and coercion'}, 'sentence_id': 3, 'trigger_index': 22}]
    event_cache[text] = {
        "sentences": sentences,
        "events": events
    }
    return {
        "events": events,
        "sentences": sentences
    }

@app.post("/predict/sentiment")  # ✅ 修正拼写
def extract_sentiment(data: TextInput):
    text = data.input_text.strip()
    if text not in event_cache:
        return {"error": "请先进行事件抽取"}

    sentences = event_cache[text]["sentences"]
    events = event_cache[text]["events"]
    causal_pairs = event_cache[text]["causal_pairs"]
    # 构造包含 offsets 的句子结构
    sentence_infos = []
    for tokens in sentences:
        text = ""
        offsets = []
        for i, tok in enumerate(tokens):
            if i > 0:
                text += " "
            start = len(text)
            text += tok
            end = len(text)
            offsets.append((start, end))
        sentence_infos.append({"sentence": text, "offsets": offsets, "events": []})

    # 插入事件并添加准确的 from/to 索引
    for e in events:
        sid = e["sentence_id"]
        trig_idx = e["trigger_index"]
        event_data = e["event"]
        start, end = sentence_infos[sid]["offsets"][trig_idx]
        new_event = {
            **event_data,
            "sentence_id": sid,
            "trigger_index": trig_idx,
            "from": start,
            "to": end
        }
        sentence_infos[sid]["events"].append(new_event)

    # 输出最终结果结构
    result = [
        {
            "text": info["sentence"],
            "events": info["events"]
        }
        for info in sentence_infos if info["events"]
    ]
    predictions, output_json = generate(result)
    output_json = attach_sentiment_to_filtered_events(output_json, predictions)
    enriched_events = sync_sentiment_back_to_events(events, result, predictions)
    # 添加 trigger_index 和 sentence_id 到 causal_pairs
    for pair in causal_pairs:
        for key in ['e1', 'e2']:
            idx = pair['i'] if key == 'e1' else pair['j']
            event = events[idx]
            pair[key]['trigger_index'] = event["trigger_index"]
            pair[key]['sentence_id'] = event["sentence_id"]

    return {
        "output_json": output_json,  # 每个句子和其事件 + 情感
        "enriched_events": enriched_events,  # 所有事件 + 情感（扁平）
        "causal_pairs": causal_pairs  # 加上因果对
    }
from typing import List
class EventItem(BaseModel):
    s: str
    v: str
    o: str = None
    p: str = None


class PredictRequest(BaseModel):
    events: List[EventItem]
    candidates: List[EventItem]
@app.post("/predict/event")
def predict_event(req: PredictRequest):
    # 模拟预测逻辑
    def to_text(event):
        parts = [event.s, event.v]
        if event.o:
            parts.append(event.o)
        if event.p:
            parts.append(event.p)
        return " ".join(parts)

    events = [to_text(e) for e in req.events]
    candidates = [to_text(c) for c in req.candidates]
    retrieved_triples = RAG(events)
    pr = f"""
### Task: 
You are an expert in **event reasoning and causal inference**. Your goal is to analyze a sequence of events and **logically infer the most plausible next step**. Carefully evaluate the **temporal, causal, and logical relationships** between events to determine the best choice.

To support your reasoning, you are given **retrieved event knowledge** from a large-scale commonsense knowledge graph (e.g., **ASER**, **ATOMIC**) that describes typical causal or temporal transitions between events.  
Use this retrieved information to **enhance your inference** and guide your step-by-step reasoning.

---

### **Guidelines:**
- Events are separated by a period (`.`).
- Consider how each candidate event aligns with the sequence in terms of **time order, causality, and logical coherence**.
- Use a **step-by-step reasoning process (Chain-of-Thought)** to gradually eliminate incorrect choices and justify why one event is the most appropriate.
- **Incorporate retrieved commonsense event relations** to support your analysis.
- Ensure your explanation follows a **progressive reasoning structure**, where each step builds upon the previous one, leading naturally to a well-supported conclusion.

---

### **Context Events:**  
{'. '.join(events)}.

---

### **Candidate Events:**  
{'. '.join(candidates)}.

---

### **Retrieved Knowledge (from ASER/ATOMIC):**
{chr(10).join(f"- {triple}" for triple in retrieved_triples)}

---

### Instruction:
Analyze the given **context events**, **candidate events**, and the **retrieved knowledge**, then logically deduce the most plausible next event.  
Wrap your final answer inside `<answer></answer>`.
"""
    result = predict(pr)
#     a = """**Step-by-Step Reasoning:**
#
# 1. **Identify the Sequence of Events:**
#    - The context events depict a group of individuals engaging in various activities: running, eating, reading, playing, singing, writing, swimming, and cooking. This suggests a diverse group with varied interests and activities.
#
# 2. **Analyze the Temporal and Causal Relationships:**
#    - The events are presented in a chronological order, with each action preceding the next. For example, John runs the marathon, followed by Mary eating pizza, and so on. This creates a logical progression of activities.
#
# 3. **Evaluate the Candidate Events:**
#    - **Sara runs in the park:** This event aligns with the theme of running, which is already present in John’s activities. However, it introduces a new character (Sara) and a new location (park), which may not fit the existing group’s dynamics.
#    - **Jake studies math:** This event introduces a new activity (studying math) that doesn’t directly relate to the previous events. It lacks a clear causal or temporal connection.
#    - **Tom drives a car:** This event introduces a new action (driving a car), which doesn’t align with the previous activities. It also introduces a new character (Tom) and doesn’t fit the established pattern.
#    - **Lily paints a picture:** This event introduces a new activity (painting) and a new character (Lily). It doesn’t directly relate to the previous events and disrupts the established sequence.
#    - **Henry dances every night:** This event introduces a new activity (dancing) and a new character (Henry). It fits well with the previous activities, as dancing is a common activity that could follow swimming or cooking.
#
# 4. **Assess Logical Coherence:**
#    - The context events suggest a group of individuals engaging in various activities, with a focus on physical and social interactions. Dancing is a natural continuation of this theme, as it involves physical activity and social interaction, which aligns with the previous events.
#
# 5. **Conclusion:**
#    - Among the candidate events, **Henry dances every night** is the most plausible next step. It maintains the logical and temporal coherence of the sequence, aligns with the established activities, and introduces a new character and activity that fits the group’s dynamics.
#
# **Final Answer:**
# 'Henry dances every night' is the most appropriate event because it logically follows the sequence of activities, maintains the established theme of physical and social interactions, and introduces a new character and activity that fits the group’s dynamics.
# <answer>Henry dances every night</answer>"""
    return {
        "result": result["response"]
        # result["response"]
    }


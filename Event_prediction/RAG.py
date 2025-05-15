
def build_causal_event_sentences(events, sentences, causal_pairs):
    # 用于快速查找：因果对中出现的事件（以元组方式标识）
    def event_key(event):
        return (event["s"], event["v"], event["o"], event["p"])

    causal_event_set = set()
    for pair in causal_pairs:
        causal_event_set.add(event_key(pair["e1"]))
        causal_event_set.add(event_key(pair["e2"]))

    # 构建句子 + offsets + events（仅限因果事件）
    result = []
    for sid, tokens in enumerate(sentences):
        text = ""
        offsets = []
        for i, tok in enumerate(tokens):
            if i > 0:
                text += " "
            start = len(text)
            text += tok
            end = len(text)
            offsets.append((start, end))

        # 收集当前句子的所有因果事件
        event_list = []
        for e in events:
            if e["sentence_id"] != sid:
                continue
            key = event_key(e["event"])
            if key in causal_event_set:
                trig_idx = e["trigger_index"]
                start, end = offsets[trig_idx]
                new_event = {
                    **e["event"],
                    "sentence_id": sid,
                    "trigger_index": trig_idx,
                    "from": start,
                    "to": end
                }
                event_list.append(new_event)

        if event_list:
            result.append({
                "text": text,
                "events": event_list
            })

    return result


def attach_sentiment_to_filtered_events(output_json, data):
    sentiment_map = {0: "positive", 1: "negative", 2: "neutral"}

    # 计算预测数是否与events数一致
    total_events = sum(len(entry["events"]) for entry in output_json)
    assert total_events == len(data), f"预测数量({len(data)})与事件数量({total_events})不一致"

    idx = 0
    for entry in output_json:
        for event in entry["events"]:
            event["sentiment"] = sentiment_map.get(data[idx], "unknown")
            idx += 1

    return output_json

def sync_sentiment_back_to_events(events, result, data):
    # Step 1: 构建 result 中的事件到 sentiment 的映射
    sentiment_map = {0: "positive", 1: "negative", 2: "neutral"}
    event_to_sentiment = {}
    idx = 0

    for item in result:
        for ev in item["events"]:
            key = (
                ev["s"], ev["v"], ev["o"], ev["p"],
                ev["sentence_id"], ev["trigger_index"]
            )
            event_to_sentiment[key] = sentiment_map.get(data[idx], "unknown")
            idx += 1

    # Step 2: 将 sentiment 同步写回原始 events 中
    for e in events:
        ev = e["event"]
        key = (
            ev["s"], ev["v"], ev["o"], ev["p"],
            e["sentence_id"], e["trigger_index"]
        )
        if key in event_to_sentiment:
            ev["sentiment"] = event_to_sentiment[key]
        else:
            ev["sentiment"] = None  # 不是因果事件对中的，设为 None 或不设

    return events

def build_event_sentences(events, sentences):
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
import sqlite3

def check_tables(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    # 查询所有表名
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("Tables in database:", tables)
    conn.close()


def check_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查 Relations 表的字段
    cursor.execute("PRAGMA table_info(Relations)")
    rel_columns = cursor.fetchall()
    print("Relations 表字段:", [col for col in rel_columns])  # 输出字段名

    # 检查 Eventualities 表的字段
    cursor.execute("PRAGMA table_info(Eventualities)")
    ev_columns = cursor.fetchall()
    print("Eventualities 表字段:", [col for col in ev_columns])

    conn.close()


import sqlite3

import sqlite3

def extract_aser_causal_triples(
    db_path,
    output_path="aser_causal_triples.txt",
    prob_threshold=0.7
):
    """
    从 ASER SQLite 中提取因果相关的三元组：Reason, Result, Condition
    """
    causal_relations = ['Reason', 'Result', 'Condition']

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 读取所有事件
    cursor.execute("SELECT _id, words FROM Eventualities")

    id2event = {row[0]: row[1] for row in cursor.fetchall() if row[1]}

    # 读取所有关系行
    cursor.execute("SELECT * FROM Relations")
    rows = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]

    print(f"🔍 共读取 Relations 表中的 {len(rows)} 条记录")

    triples = []
    for row in rows:
        row_dict = dict(zip(columns, row))
        e1_id = row_dict["event1_id"]
        e2_id = row_dict["event2_id"]

        if e1_id not in id2event or e2_id not in id2event:
            continue

        e1_text = id2event[e1_id].strip()
        e2_text = id2event[e2_id].strip()

        for rel in causal_relations:
            prob = row_dict.get(rel)
            if prob is not None and prob >= prob_threshold:
                triple = f"{e1_text} → {rel.lower()} → {e2_text}"
                triples.append(triple)

    print(f"✅ 提取因果三元组：{len(triples)} 条（prob ≥ {prob_threshold}）")

    with open(output_path, "w", encoding="utf-8") as f:
        for t in triples:
            f.write(t + "\n")

    print(f"🎉 已保存至 {output_path}")

    conn.close()


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from tqdm import tqdm
import os
from app.utils.retriever import AserRetriever
from app.utils.reranker import TripleReranker
def load_model(model_path_or_name):
    if os.path.exists(model_path_or_name):
        print("📦 正在从本地加载模型...")
        return SentenceTransformer(model_path_or_name)
    else:
        print("🌐 正在从 HuggingFace 下载模型...")
        return SentenceTransformer(model_path_or_name)

def build_faiss_index_for_CT(
    triple_path="causal_event_pairs.txt",
    model_name=r"D:\NLP\project_for_system\my_backend\app\bge-small-en-v1.5",  # 可替换为你自己的
    index_save_path="faiss_index/ct_causal.index",
    text_save_path="faiss_index/ct_causal_texts.npy",
    normalize=True
):
    """
    构建三元组向量索引库（用于RAG语义检索）
    """
    # 加载三元组
    with open(triple_path, 'r', encoding='utf-8') as f:
        triples = [line.strip() for line in f if line.strip()]

    print(f"📄 共加载三元组: {len(triples)} 条")

    # 加载模型
    print(f"🧠 加载编码模型: {model_name}")
    model = SentenceTransformer(model_name)

    # 向量编码
    print("🔍 正在编码三元组...")
    embeddings = model.encode(triples, show_progress_bar=True, normalize_embeddings=normalize)
    embeddings = np.array(embeddings).astype("float32")

    # 构建 FAISS 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) if normalize else faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ 构建完毕！索引向量数: {index.ntotal}")

    # 保存
    import os
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, index_save_path)
    np.save(text_save_path, np.array(triples))

    print(f"📦 索引保存至: {index_save_path}")
    print(f"📦 原文保存至: {text_save_path}")
def build_faiss_index(
    triple_path="aser_triples.txt",
    model_name=r"D:\NLP\project_for_system\my_backend\app\bge-small-en-v1.5",  # 可替换为你自己的
    index_save_path="faiss_index/aser_causal.index",
    text_save_path="faiss_index/aser_causal_texts.npy",
    normalize=True
):
    """
    构建三元组向量索引库（用于RAG语义检索）
    """
    # 加载三元组
    with open(triple_path, 'r', encoding='utf-8') as f:
        triples = [line.strip() for line in f if line.strip()]

    print(f"📄 共加载三元组: {len(triples)} 条")

    # 加载模型
    print(f"🧠 加载编码模型: {model_name}")
    model = SentenceTransformer(model_name)

    # 向量编码
    print("🔍 正在编码三元组...")
    embeddings = model.encode(triples, show_progress_bar=True, normalize_embeddings=normalize)
    embeddings = np.array(embeddings).astype("float32")

    # 构建 FAISS 索引
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim) if normalize else faiss.IndexFlatL2(dim)
    index.add(embeddings)
    print(f"✅ 构建完毕！索引向量数: {index.ntotal}")

    # 保存
    import os
    os.makedirs("faiss_index", exist_ok=True)
    faiss.write_index(index, index_save_path)
    np.save(text_save_path, np.array(triples))

    print(f"📦 索引保存至: {index_save_path}")
    print(f"📦 原文保存至: {text_save_path}")
import re
import networkx as nx
from typing import List
from sentence_transformers import SentenceTransformer
def is_bad_event(text: str):
    stopwords = {"be", "do", "know", "say", "have", "make", "see"}
    tokens = text.lower().split()
    return (
        len(tokens) <= 2 or
        any(token in stopwords for token in tokens) or
        re.match(r"^\s*(he|she|it|they|i|you)\s*$", text.strip().lower())
    )

def filter_bad_triples(triples):
    filtered = []
    for t in triples:
        try:
            e1, rel, e2 = t.split("→")
            if not is_bad_event(e1) and not is_bad_event(e2):
                filtered.append(t.strip())
        except:
            continue
    return filtered

def load_graph_from_triples(triple_path: str) -> nx.DiGraph:
    """
    根据三元组文件构建事件因果有向图 G(event1 → event2)
    每条边保存了因果关系 label
    """
    G = nx.DiGraph()
    with open(triple_path, 'r', encoding='utf-8') as f:
        for line in f:
            if "→" not in line:
                continue
            try:
                e1, rel, e2 = [s.strip() for s in line.split("→")]
                G.add_edge(e1, e2, label=rel)
            except Exception as e:
                print(f"[WARN] Parse failed: {line.strip()} | {e}")
    return G


def semantic_path_beam_search(
    G: nx.DiGraph,
    start_nodes: List[str],
    query: str,
    encoder: SentenceTransformer,
    max_hops: int = 5,
    beam_width: int = 5,
    top_k_paths: int = 10,
    sim_threshold: float = 0.3
) -> List[List[str]]:
    """
    使用 Beam Search 进行多跳语义路径搜索
    """
    query_emb = encoder.encode([query], normalize_embeddings=True)[0]
    path_candidates = []

    def path_score(path_embs):
        return np.mean([np.dot(e, query_emb) for e in path_embs])

    for start in start_nodes:
        if start not in G:
            continue

        # 初始化 beam 为 [[start]] 路径列表
        beam = [[start]]

        for _ in range(max_hops):
            new_beam = []
            for path in beam:
                last_node = path[-1]
                for succ in G.successors(last_node):
                    if succ in path:
                        continue  # 避免环
                    new_path = path + [succ]
                    embs = encoder.encode(new_path, normalize_embeddings=True)
                    score = path_score(embs)
                    if score >= sim_threshold:
                        new_beam.append((new_path, score))
            if not new_beam:
                break
            # 仅保留前 beam_width 条路径
            new_beam.sort(key=lambda x: x[1], reverse=True)
            beam = [p for p, _ in new_beam[:beam_width]]

        # 添加所有 beam 中的路径
        path_candidates.extend(beam)

    # 全局筛选 top_k_paths
    scored_paths = [
        (path, path_score(encoder.encode(path, normalize_embeddings=True)))
        for path in path_candidates
    ]
    scored_paths.sort(key=lambda x: x[1], reverse=True)
    top_paths = [p for p, _ in scored_paths[:top_k_paths]]
    return top_paths


def deduplicate_and_filter_paths(
    paths: List[List[str]],
    encoder: SentenceTransformer,
    min_len: int = 3,
    min_diversity: float = 0.2
) -> List[List[str]]:
    """
    路径去重 + 简单的多样性过滤
    """
    seen = set()
    good_paths = []

    def diversity_score(path_embs):
        return np.mean([1 - np.dot(path_embs[i], path_embs[i+1]) for i in range(len(path_embs)-1)])

    for path in paths:
        if len(path) < min_len:
            continue
        t = tuple(path)
        if t in seen:
            continue
        seen.add(t)
        embs = encoder.encode(path, normalize_embeddings=True)
        if diversity_score(embs) >= min_diversity:
            good_paths.append(path)
    return good_paths


def print_paths(paths: List[List[str]], encoder: SentenceTransformer, query: str):
    """
    打印路径及其与 query 的平均相似度
    """
    query_emb = encoder.encode([query], normalize_embeddings=True)[0]
    for path in paths:
        embs = encoder.encode(path, normalize_embeddings=True)
        score = np.mean([np.dot(e, query_emb) for e in embs])
        print(" → ".join(path), f"(score={score:.3f})")

# build_faiss_index_for_CT()
# # 示例入口
# if __name__ == '__main__':
#     from sentence_transformers import SentenceTransformer
#
#     # 配置路径
#     TRIPLE_PATH = "aser_triples.txt"
#     MODEL_PATH = r"D:\NLP\project_for_system\my_backend\app\bge-small-en-v1.5"
#
#     # 加载图谱和模型
#     G = load_graph_from_triples(TRIPLE_PATH)
#
#     # 输入事件上下文
#     context_events = [
#         "friends knew simpson",
#         "simpson became famous",
#         "simpson was convicted",
#         "law applied to simpson"
#     ]
#     query = ". ".join(context_events)
#     retriever = AserRetriever(
#         index_path="faiss_index/aser_causal.index",
#         text_path="faiss_index/aser_causal_texts.npy",
#         model_name=r"D:\NLP\project_for_system\my_backend\app\bge-small-en-v1.5"  # 或你自己的本地模型路径
#     )
#
#     context_events = [
#         "friends knew simpson",
#         "simpson became famous",
#         "simpson was convicted",
#         "law applied to simpson"
#     ]
#     index = faiss.read_index("faiss_index/aser_causal.index")
#     texts = np.load("faiss_index/aser_causal_texts.npy", allow_pickle=True).tolist()
#     # 1. 加载 ASER 全图（多跳结构支持）
#     graph = load_graph_from_triples("aser_triples.txt")
#
#     # 1. 原始检索
#     retrieved,encoder = retriever.retrieve(context_events, top_k=20)
#
#     # # 2. 过滤低质量事件
#     # cleaned = filter_bad_triples(retrieved)
#
#     # 3. 使用 reranker 进行语义排序
#     reranker = TripleReranker()
#     top_reranked = reranker.rerank(query, retrieved, top_k=5)
#     seed_events = set()
#     for triple in top_reranked:
#         e1, _, e2 = [s.strip() for s in triple.split("→")]
#         seed_events.add(e1)
#         seed_events.add(e2)
#     # 执行路径搜索
#     paths = semantic_path_beam_search(
#         G=G,
#         start_nodes=seed_events,
#         query=query,
#         encoder=encoder,
#         max_hops=5,
#         beam_width=5,
#         top_k_paths=20,
#         sim_threshold=0.3
#     )
#
#     # 路径过滤
#     cleaned = deduplicate_and_filter_paths(paths, encoder, min_len=2, min_diversity=0.2)
#
#     print("Top semantic paths:")
#     print_paths(cleaned, encoder, query)

def RAG(context_events):
    retriever = AserRetriever(
        index_path=r"D:\NLP\project_for_system\my_backend\app\utils\faiss_index\ct_causal.index",
        text_path=r"D:\NLP\project_for_system\my_backend\app\utils\faiss_index\ct_causal_texts.npy",
        model_name=r"D:\NLP\project_for_system\my_backend\app\bge-small-en-v1.5"  # 或你自己的本地模型路径
    )
    query = ". ".join(context_events)

    # 1. 原始检索
    retrieved, encoder = retriever.retrieve(context_events, top_k=50)


    # 2. 使用 reranker 进行语义排序
    reranker = TripleReranker()
    top_reranked = reranker.rerank(query, retrieved, top_k=8)
    return top_reranked
# if __name__ == '__main__':
#     retriever = AserRetriever(
#         index_path=r"D:\NLP\project_for_system\my_backend\app\utils\faiss_index\aser_causal.index",
#         text_path=r"D:\NLP\project_for_system\my_backend\app\utils\faiss_index\aser_causal_texts.npy",
#         model_name=r"D:\NLP\project_for_system\my_backend\app\bge-small-en-v1.5"  # 或你自己的本地模型路径
#     )
#     context_events = ["trump"]
#     query = ". ".join(context_events)
#
#     # 1. 原始检索
#     retrieved, encoder = retriever.retrieve(context_events, top_k=50)
#
#     # # 2. 过滤低质量事件
#     # cleaned = filter_bad_triples(retrieved)
#
#     # 3. 使用 reranker 进行语义排序
#     reranker = TripleReranker()
#     top_reranked = reranker.rerank(query, retrieved, top_k=8)

#
#     # check_tables('../output/KG.db')
#     # check_schema('../output/KG.db')
#     # extract_aser_causal_triples('../output/KG.db', "aser_triples.txt", prob_threshold=0.75)
#     # model = load_model(r"D:\NLP\project_for_system\my_backend\app\bge-small-en-v1.5")
#
#     # build_faiss_index()
#     retriever = AserRetriever(
#         index_path="faiss_index/aser_causal.index",
#         text_path="faiss_index/aser_causal_texts.npy",
#         model_name=r"D:\NLP\project_for_system\my_backend\app\bge-small-en-v1.5"  # 或你自己的本地模型路径
#     )
#
#     context_events = [
#         "friends knew simpson",
#         "simpson became famous",
#         "simpson was convicted",
#         "law applied to simpson"
#     ]
#     index = faiss.read_index("faiss_index/aser_causal.index")
#     texts = np.load("faiss_index/aser_causal_texts.npy", allow_pickle=True).tolist()
#     # 1. 加载 ASER 全图（多跳结构支持）
#     graph = load_graph_from_triples("aser_triples.txt")
#     query = ". ".join(context_events)
#
#     # 1. 原始检索
#     retrieved,encoder = retriever.retrieve(context_events, top_k=50)
#
#     # # 2. 过滤低质量事件
#     # cleaned = filter_bad_triples(retrieved)
#
#     # 3. 使用 reranker 进行语义排序
#     reranker = TripleReranker()
#     top_reranked = reranker.rerank(query, retrieved, top_k=8)
#     print(top_reranked)
#     seed_events = set()
#     for triple in top_reranked:
#         e1, _, e2 = [s.strip() for s in triple.split("→")]
#         seed_events.add(e1)
#         seed_events.add(e2)
#     from itertools import chain
#
#     paths = semantic_random_walk_paths_from_global_index_fixed(
#         G=graph,
#         start_nodes=seed_events,
#         query=query,
#         encoder=encoder,
#         faiss_index=index,
#         faiss_texts=texts,
#         max_hops=5,
#         walks_per_node=10,
#         top_k_per_hop=10,
#         sim_threshold=0.2
#     )
#
#     cleaned_paths = deduplicate_and_filter_paths(
#         paths=paths,
#         encoder=encoder,
#         min_len=3,
#         min_diversity=0.2
#     )
#
#     print("随机游走生成的路径数量:", len(paths))
#     for p in paths[:5]:
#         print(" -> ".join(p))



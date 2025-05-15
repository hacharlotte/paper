from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class AserRetriever:
    def __init__(self,
                 index_path="faiss_index/aser_causal.index",
                 text_path="faiss_index/aser_causal_texts.npy",
                 model_name="BAAI/bge-small-en-v1.5",  # 可换成你的本地模型路径
                 normalize=True):
        """
        初始化检索器，加载索引、三元组和向量模型
        """
        print("📥 加载外部知识...")
        self.triples = np.load(text_path, allow_pickle=True).tolist()

        print("🧠 加载模型中...")
        self.model = SentenceTransformer(model_name)

        print("📦 加载向量索引...")
        self.index = faiss.read_index(index_path)
        self.normalize = normalize

        # print(f"✅ 初始化完成，共载入 {len(self.triples)} 条三元组")
        print(f"✅ 初始化完成")

    def retrieve(self, event_list, top_k=5):
        """
        输入：一组事件（字符串）
        输出：最相关的 Top-K 三元组
        """
        # 将事件拼接成一段文本：提高上下文感知能力
        query = ". ".join(event_list)
        query_vec = self.model.encode([query], normalize_embeddings=self.normalize).astype("float32")

        D, I = self.index.search(query_vec, top_k)
        return [self.triples[i] for i in I[0]], self.model

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class TripleReranker:
    def __init__(self, model_name=r"D:\NLP\project_for_system\my_backend\app\bge-reranker-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    def rerank(self, query: str, triples: list, top_k=5):
        scored = []
        for triple in triples:
            inputs = self.tokenizer(query, triple, truncation=True, return_tensors="pt")
            with torch.no_grad():
                score = self.model(**inputs).logits[0].item()
            scored.append((score, triple))
        scored.sort(reverse=True)
        return [t for _, t in scored[:top_k]]

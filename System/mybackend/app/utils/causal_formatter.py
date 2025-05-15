class CausalFormatter:

    @staticmethod
    def format_pairs(features, predicted, events):
        causal_pairs = []
        for idx, label in enumerate(predicted):
            if label == 1:
                i, j = features.event_pairs[idx]
                e1 = events[i]["event"]
                e2 = events[j]["event"]
                causal_pairs.append({
                    "id": idx,
                    "i": i,
                    "j": j,
                    "e1": e1,
                    "e2": e2
                })
        return causal_pairs

    @staticmethod
    def paginate(pairs, page=1, page_size=20):
        start = (page - 1) * page_size
        end = start + page_size
        return {
            "total": len(pairs),
            "page": page,
            "page_size": page_size,
            "data": pairs[start:end]
        }

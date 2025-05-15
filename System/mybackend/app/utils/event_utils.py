# app/utils/event_utils.py

def extract_events(text):
    # ✨ 实际建议用你的事件抽取模型，这里只是模拟
    # 我们假设提取了两个事件
    return [
        {"s": "Barack Obama", "v": "visited", "o": "Japan", "p": None},
        {"s": "Barack Obama", "v": "made", "o": "an appeal", "p": None}
    ]

def build_event_pairs(events):
    # 构建事件对
    pairs = []
    for i in range(len(events)):
        for j in range(i+1, len(events)):
            pairs.append([events[i]['v'], events[j]['v']])
    return pairs

# app/model/causality_model.py
from app.utils.model_nopenal import bertCSRModel_nosem
from app.utils.parameter import parse_args
import torch
from app.utils.utils import transfor3to2, compute_f1, setup_seed, calculate
from transformers import BertConfig, BertTokenizer
MODEL_CLASSES = {
    'bert': (BertConfig, bertCSRModel_nosem, BertTokenizer)
}
def predict(features):
    # 这里用假的结果模拟，正常应调用你的真实模型
    args = parse_args()
    torch.cuda.empty_cache()
    args.device = None
    setup_seed(args.seed)
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    config_class, eci_model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    model = eci_model_class(args).to(args.device)
    # 加载 state_dict
    model.load_state_dict(torch.load(
        r"D:\NLP\project_for_system\my_backend\app\output\best_model_fold_2_20250105_F1_0.7201.pt",
        map_location=args.device
    ))
    model.eval()
    inputs = {'enc_input_ids': features.enc_input_ids.to(args.device),
              'enc_mask_ids': features.enc_mask_ids.to(args.device),
              'node_event': features.node_event,
              't1_pos': [features.t1_pos],
              't2_pos': [features.t2_pos],
              'target': [features.target],
              'rel_type': [features.rel_type],
              'event_pairs': features.event_pairs
              }
    loss, opt = model(**inputs)
    predicted_ = torch.argmax(opt, -1)
    predicted_ = list(predicted_.cpu().numpy())
    gold_ = [t for bt in inputs['target'] for t in bt]
    clabel = [t for bt in inputs['rel_type'] for t in bt]
    # transform the three classification results predicted by the model to Causality Existence Identification and Causality Direction Identification results
    # predicted: the Identification Result of Causality Direction(we calculate the micro-averaged results for Precision, Recall, and F1-score specifically for the CAUSE and EFFECT classes)
    gold, predicted = transfor3to2(gold_, predicted_)
    return predicted  # 暂时直接返回 pairs，假装模型预测了
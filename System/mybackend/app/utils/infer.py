import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
from swift.llm import InferEngine, InferRequest, PtEngine, RequestConfig, get_template

infer_backend = 'pt'

# 生成参数
max_new_tokens = 1024
temperature = 0
stream = True
def infer_stream(engine: InferEngine, infer_request: InferRequest):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature, stream=True)
    gen = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    print(f'query: {query}\nresponse: ', end='')
    for resp_list in gen:
        print(resp_list[0].choices[0].delta.content, end='', flush=True)
    print()

def infer(engine: InferEngine, infer_request: InferRequest):
    request_config = RequestConfig(max_tokens=max_new_tokens, temperature=temperature)
    resp_list = engine.infer([infer_request], request_config)
    query = infer_request.messages[0]['content']
    response = resp_list[0].choices[0].message.content
    return {"query" : query, "response" :response}
def predict(text):
    infer_func = infer
    last_model_checkpoint1 = r'D:\NLP\project_for_system\my_backend\app\output\v29-20250321-235313/checkpoint-924'

    # 模型
    model_id_or_path = r'D:\NLP\project_for_system\my_backend\app\deepseekv3'
    system = "<system>"
    engine = PtEngine(model_id_or_path, adapters=[last_model_checkpoint1], model_type='deepseek_r1_distill',
                      use_hf=False)
    engine.model = engine.model.to(torch.float32)
    template = get_template(engine.model.model_meta.template, engine.tokenizer, default_system=system)
    # 这里对推理引擎的默认template进行修改，也可以在`engine.infer`时进行传入
    engine.default_template = template
    query_list = [text]
    for query in query_list:
        return infer_func(engine, InferRequest(messages=[{'role': 'user', 'content': query}]))

# # 这里使用了2个infer_request来展示batch推理
# infer_requests = [
#     InferRequest(messages=[{'role': 'user', 'content': "<Event_Mask_Predictor>\nTask: You are an event predictor. Given an event sequence with some events masked, predict the missing events.\nNote: Events are separated by a period ('.').\n\nEvent Sequence: clinton amplified theme . clinton attacked proposal . accompanied clinton . clinton addressed student . <MASK> . smith called clinton lord . clinton received ovation . <MASK> . clinton noted\n\nAnswer:"}])
# ]
# resp_list = engine.infer(infer_requests, request_config)
# query0 = infer_requests[0].messages[0]['content']
# print(f'response0: {resp_list[0].choices[0].message.content}')
# print(f'response1: {resp_list[1].choices[0].message.content}')
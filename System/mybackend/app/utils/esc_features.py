class ESC_features(object):
    def __init__(self, topic_id, doc_id,
                 enc_text, enc_tokens, sentences,
                 enc_input_ids, enc_mask_ids, node_event,
                 t1_pos, t2_pos, target, rel_type, event_pairs
                 ):
        self.topic_id = topic_id
        self.doc_id = doc_id
        self.enc_text = enc_text
        self.enc_tokens = enc_tokens
        self.sentences = sentences
        self.enc_input_ids = enc_input_ids
        self.enc_mask_ids = enc_mask_ids
        self.node_event = node_event
        self.t1_pos = t1_pos
        self.t2_pos = t2_pos
        self.target = target
        self.rel_type = rel_type
        self.event_pairs = event_pairs
import argparse

def get_parameter():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--data_dir', type=str, default='data/Laptop')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--num_workers_per_loader', type=int, default=0)
    parser.add_argument('--report_frequency', type=int, default=1000)
    parser.add_argument('--save_frequency', type=int, default=2000)
    parser.add_argument('--num_class', type=int, default=3, help='Num of sentiment class.')

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='bert dim.')
    parser.add_argument('--hidden_dim', type=int, default=768)

    parser.add_argument('--input_dropout', type=float, default=0.2, help='input dropout rate.')
    parser.add_argument('--layer_dropout', type=float, default=0.1, help='layer dropout rate.')
    parser.add_argument('--attn_dropout', type=float, default=0.0, help='self-attention layer dropout rate.')
    parser.add_argument('--attn_head', type=int, default=6)
    parser.add_argument('--max_num_spans', type=int, default=1)
    parser.add_argument('--num_encoder_layer', type=int, default=3, help='Number of graph layers.')
    parser.add_argument('--log_step', type=int, default=16, help='Print log every k steps.')
    parser.add_argument('--lower', default=True, help='lowercase all words.')
    parser.add_argument('--optimizer', default='adamw', help='need parse data.')

    parser.add_argument('--learning_rate', type=float, default=2e-5, help='learning rate.')##dot:2e-10->1e-8->2e-9 sum:1e-8->2e-9
    parser.add_argument('--bert_lr', type=float, default=2e-5, help='learning rate for bert.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay rate.')

    parser.add_argument('--epoch', type=int, default=20, help='Number of total training epochs.')
    parser.add_argument('--warm_up', type=int, default=70, help='max patience in training')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size.')#####最佳为batch16，epoch20
    parser.add_argument('--label_smooth', type=float, default=0.01, help='Print log every k steps.')
    parser.add_argument('--sort_key_idx', default=0, help='sort idx')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_len', type=int, default=100)
    parser.add_argument('--grad_norm', type=float, default=5.0)
    parser.add_argument('--dep_relation_embed_dim', type=int, default=768)
    args = parser.parse_args()

    return args
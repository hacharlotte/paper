import torch
import torch.nn as nn
import math
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        位置编码层
        :param d_model: 嵌入维度
        :param max_len: 序列的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 生成一个固定的position矩阵，大小为(max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))  # 分母部分

        # 使用sin和cos函数填充位置编码
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置使用sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置使用cos

        pe = pe.unsqueeze(0)  # 增加batch维度 [1, max_len, d_model]
        self.register_buffer('pe', pe)  # 保证位置编码是常数，不会在训练中更新

    def forward(self, x):
        """
        :param x: 输入的嵌入表示 [seq_len, d_model]
        :return: 添加了位置编码后的输入
        """
        # 将位置编码添加到输入中，保证序列的长度不超过max_len
        seq_len = x.size(0)
        return x + self.pe[:, :seq_len].squeeze(0)

class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        可学习的位置编码层
        :param d_model: 嵌入维度
        :param max_len: 序列的最大长度
        """
        super(LearnablePositionalEncoding, self).__init__()

        # 位置编码矩阵，形状为(max_len, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)  # 可学习的位置嵌入
        self.register_buffer('position_ids', torch.arange(max_len).unsqueeze(0))  # 位置ID

    def forward(self, x):
        """
        :param x: 输入的嵌入表示 [batch_size, seq_len, d_model]
        :return: 添加了位置编码后的输入
        """
        seq_len = x.size(0)  # 获取序列长度
        position_ids = self.position_ids[:, :seq_len]  # 获取对应的position_ids
        position_embeddings = self.position_embeddings(position_ids).squeeze(0)  # 获取对应位置的嵌入
        return x + position_embeddings  # 加入位置编码
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, ff_hid_dim, max_len, dropout=0.1):
        """
        Transformer模型
        :param input_dim: 输入的特征维度
        :param d_model: 模型的嵌入维度
        :param num_heads: 多头自注意力的头数
        :param num_layers: Transformer Encoder 层的数量
        :param ff_hid_dim: 前馈网络的隐藏维度
        :param dropout: Dropout比率
        """
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 位置编码层
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        # Transformer Encoder层
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=ff_hid_dim,
                                                    dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)

        # 输出层
        self.fc_out = nn.Linear(d_model, input_dim)

    def forward(self, x):
        """
        前向传播
        :param src: 输入序列 (batch_size, seq_len, d_model)
        :return: Transformer 输出 (batch_size, seq_len, d_model)
        """

        # 对输入进行嵌入，加入位置编码
        x = x * math.sqrt(self.d_model)  # 嵌入缩放
        # 获取位置编码并加到输入上
        x = self.positional_encoding(x)
        x = x.unsqueeze(0)
        # Transformer Encoder
        transformer_output, attention_weights = self.transformer_encoder_with_attention(x)
        # * Average Multi-head Attention matrixes
        adj_ag = None
        for i in range(self.num_layers):
            if adj_ag is None:
                adj_ag = attention_weights[i]
            else:
                adj_ag += attention_weights[i]
        adj_ag /= self.num_layers
        # 输出层
        output = self.fc_out(transformer_output)
        output = output.squeeze(0)

        return output, adj_ag

    def transformer_encoder_with_attention(self, x):
        """
        获取Transformer Encoder的输出和多头注意力矩阵
        :param x: 输入序列 (batch_size, seq_len, d_model)
        :return: 输出序列 (batch_size, seq_len, d_model), 注意力矩阵 (num_heads, seq_len, seq_len)
        """
        all_attention_weights = []

        # 遍历每一层
        for layer in self.transformer_encoder.layers:
            # 获取MultiheadAttention层
            multihead_attention = layer.self_attn

            # 获取注意力权重，MultiheadAttention的forward方法返回的是(attention_output, attention_weights)
            # 其中 attention_weights 形状是 (batch_size, num_heads, seq_len, seq_len)
            attention_output, attention_weights = multihead_attention(x, x, x)
#             print("attention_weights",attention_weights.shape)
#         # 更新输入 x 为该层的输出（即，attention_output）
#             x = layer.norm1(x + attention_output)  # 残差连接 + 归一化
#                     # 通过前馈网络 (Feedforward Network) 进一步学习
#             feedforward_output = layer.linear2(layer.dropout(layer.activation(layer.linear1(x))))
#             x = layer.norm2(x + feedforward_output)  # 残差连接 + 归一化
            # 如果想保留每个头的注意力权重
            all_attention_weights.append(attention_weights)

        # 合并所有头的注意力权重
        # all_attention_weights是一个list，每个元素的形状是 [batch_size, num_heads, seq_len, seq_len]
        # 对所有的层进行合并，得到一个最终的注意力矩阵
        attention_weights = torch.stack(all_attention_weights, dim=0)  # 形状 [num_layers, batch_size, num_heads, seq_len, seq_len]

        return x, attention_weights
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.t5.modeling_t5 import T5LayerNorm

class SyntaxAttentionGAT2(nn.Module):
    def __init__(self, in_channel=768, out_channel=256, label_num=6, num_dep_labels=45, dep_embed_dim=64):
        super(SyntaxAttentionGAT2, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.label_num = label_num
        self.num_dep_labels = num_dep_labels
        self.dep_embed_dim = dep_embed_dim

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.dense1 = nn.Linear(self.in_ch, self.in_ch)
        self.dense2 = nn.Linear(self.in_ch, self.out_ch)

        # 依存类别嵌入层
        self.dep_embedding = nn.Embedding(num_dep_labels, dep_embed_dim)

        layer_norm_eps = 1e-12
        self.norm1 = T5LayerNorm(in_channel, layer_norm_eps)
        self.norm2 = T5LayerNorm(self.out_ch, layer_norm_eps)

    def forward(self, table, a_simi, t_simi, syntax_matrix):  # [B,L,L,dim], [B,L], [B,L], [B,L,L]
        batch, seq, seq, dim = table.shape
        syntax_matrix = syntax_matrix[:, :seq, :seq]  # 截取句法依存矩阵的有效部分

        # 将句法依存矩阵转化为注意力权重
        syntax_attention = self._syntax_to_attention(syntax_matrix)  # [B,L,L,1]

        # 原有的 a_simi 和 t_simi 操作
        a_simi2 = a_simi.unsqueeze(2).expand(batch, seq, seq).unsqueeze(3)  # [B,L,L,1]
        t_simi2 = t_simi.unsqueeze(1).expand(batch, seq, seq).unsqueeze(3)  # [B,L,L,1]

        h0 = table  # 初始输入
        h1 = self.gether(h0, a_simi2, t_simi2)  # 保留原有的局部邻域操作
        h1 = h1 + h0 * syntax_attention  # 加入句法依存注意力权重
        h1 = self.relu(self.norm1(self.dense1(h1)))  # 全连接 + 归一化 + 激活

        h2 = self.gether(h1, a_simi2, t_simi2)  # 保留原有的局部邻域操作
        h2 = h2 + h1 * syntax_attention  # 加入句法依存注意力权重
        h2 = self.relu(self.norm2(self.dense2(h2)))  # 全连接 + 归一化 + 激活

        return h2

    def gether(self, h, a_simi, t_simi):
        """
        保留原有的局部邻域操作
        :param h: 输入特征 [B,L,L,dim]
        :param a_simi: 注意力权重 [B,L,L,1]
        :param t_simi: 注意力权重 [B,L,L,1]
        :return: 聚合后的特征 [B,L,L,dim]
        """
        batch, seq, seq, dim = h.shape

        # 原有的局部邻域操作
        padding_y = torch.zeros([batch, seq, 1, dim]).cuda()  # [B,L,1,dim]
        h_y = h * t_simi
        h_left = torch.cat([padding_y, h_y], dim=2)[:, :, :-1, :]  # [B,L,L,dim]
        h_right = torch.cat([h_y, padding_y], dim=2)[:, :, 1:, :]

        padding_x = torch.zeros([batch, 1, seq, dim]).cuda()  # [B,1,L,dim]
        h_x = h * a_simi
        h_up = torch.cat([padding_x, h_x], dim=1)[:, :-1, :, :]  # [B,L,L,dim]
        h_down = torch.cat([h_x, padding_x], dim=1)[:, 1:, :, :]

        h = h + h_left + h_right + h_up + h_down
        return h

    def _syntax_to_attention(self, syntax_matrix):
        """
        将句法依存矩阵转化为注意力权重
        :param syntax_matrix: 句法依存矩阵 [B,L,L]
        :return: 注意力权重 [B,L,L,1]
        """
        # 将依存类别编号映射为嵌入向量
        dep_embeddings = self.dep_embedding(syntax_matrix)  # [B,L,L,dep_embed_dim]

        # 计算注意力权重（例如，通过点积）
        attention_weights = torch.sum(dep_embeddings, dim=-1, keepdim=True)  # [B,L,L,1]

        # 归一化（softmax）
        attention_weights = F.softmax(attention_weights, dim=2)  # [B,L,L,1]

        return attention_weights
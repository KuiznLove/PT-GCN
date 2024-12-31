import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5LayerNorm
import torch.nn.functional as F
from .myutils import BertLinear, BertLayerNorm, gelu

class SyntaxGAT(torch.nn.Module):
    def __init__(self, in_channel=768, out_channel=256, label_num=6):
        super(SyntaxGAT, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.label_num = label_num

        # 原有组件
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
        self.dense1 = torch.nn.Linear(self.in_ch, self.in_ch)
        self.dense2 = torch.nn.Linear(self.in_ch, self.out_ch)

        # 新增句法依存相关组件
        self.dep_embedding = nn.Embedding(44, 6)
        self.dep_attention = nn.Linear(6, 1)
        self.syntax_gate = nn.Linear(self.in_ch * 2, self.in_ch)

        layer_norm_eps = 1e-12
        self.norm1 = T5LayerNorm(in_channel, layer_norm_eps)
        self.norm2 = T5LayerNorm(self.out_ch, layer_norm_eps)

    def forward(self, table, a_simi, t_simi, dep_matrix, dep_types):
        # table: [B,L,L,dim]
        # a_simi, t_simi: [B,L]
        # dep_matrix: [B,L,L] 依存关系邻接矩阵
        # dep_types: [B,L,L] 依存关系类型

        batch, seq, seq, dim = table.shape

        # 扩展相似度矩阵
        a_simi2 = a_simi.unsqueeze(2).expand(batch, seq, seq).unsqueeze(3)
        t_simi2 = t_simi.unsqueeze(1).expand(batch, seq, seq).unsqueeze(3)

        # 处理依存关系
        dep_embeds = self.dep_embedding(dep_types)  # [B,L,L,dep_dim]
        dep_weights = self.dep_attention(dep_embeds)  # [B,L,L,1]
        dep_weights = torch.sigmoid(dep_weights) * dep_matrix.unsqueeze(-1)

        h0 = table
        # 第一层GAT
        h1_semantic = self.gether(h0, a_simi2, t_simi2)
        h1_syntax = self.syntax_gether(h0, dep_weights)
        h1 = self.feature_fusion(h1_semantic, h1_syntax)
        h1 = self.relu(self.norm1(self.dense1(h1 + h0)))

        # 第二层GAT
        h2_semantic = self.gether(h1, a_simi2, t_simi2)
        h2_syntax = self.syntax_gether(h1, dep_weights)
        h2 = self.feature_fusion(h2_semantic, h2_syntax)
        h2 = self.relu(self.norm2(self.dense2(h2 + h1)))

        return h2

    def syntax_gether(self, h, dep_weights):
        """基于句法依存的特征聚合"""
        batch, seq, seq, dim = h.shape

        # 依存关系加权
        h_dep = h * dep_weights

        # 考虑入边和出边
        h_in = torch.sum(h_dep, dim=2)  # 聚合入边信息
        h_out = torch.sum(h_dep, dim=1)  # 聚合出边信息

        # 扩展维度以匹配原始特征
        h_in = h_in.unsqueeze(2).expand(-1, -1, seq, -1)
        h_out = h_out.unsqueeze(1).expand(-1, seq, -1, -1)

        return h_in + h_out

    def feature_fusion(self, semantic_feat, syntax_feat):
        """融合语义特征和句法特征"""
        # 计算融合门控值
        combined = torch.cat([semantic_feat, syntax_feat], dim=-1)
        gate = torch.sigmoid(self.syntax_gate(combined))

        # 动态加权融合
        fused_feat = gate * semantic_feat + (1 - gate) * syntax_feat
        return fused_feat

    def gether(self, h, a_simi, t_simi):
        """语义特征聚合"""
        batch, seq, seq, dim = h.shape

        # 水平方向特征聚合
        padding_y = torch.zeros([batch, seq, 1, dim]).cuda()
        h_y = h * t_simi
        h_left = torch.cat([padding_y, h_y], dim=2)[:, :, :-1, :]
        h_right = torch.cat([h_y, padding_y], dim=2)[:, :, 1:, :]

        # 垂直方向特征聚合
        padding_x = torch.zeros([batch, 1, seq, dim]).cuda()
        h_x = h * a_simi
        h_up = torch.cat([padding_x, h_x], dim=1)[:, :-1, :, :]
        h_down = torch.cat([h_x, padding_x], dim=1)[:, 1:, :, :]

        h = h + h_left + h_right + h_up + h_down
        return h

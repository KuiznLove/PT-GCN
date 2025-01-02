import torch
from transformers.models.t5.modeling_t5 import T5LayerNorm

class SyntaxAttentionGAT(torch.nn.Module):
    def __init__(self, in_channel=768, out_channel=256, num_heads=8):
        super(SyntaxAttentionGAT, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()

        # 原有的变换层
        self.dense1 = torch.nn.Linear(self.in_ch, self.in_ch)
        self.dense2 = torch.nn.Linear(self.in_ch, self.out_ch)

        # 句法依存相关层
        self.dep_proj = torch.nn.Linear(1, 1)  # 句法依存权重投影
        self.dep_gate = torch.nn.Linear(in_channel * 2, 1)  # 控制句法信息融合的门控

        layer_norm_eps = 1e-12
        self.norm1 = T5LayerNorm(in_channel, layer_norm_eps)
        self.norm2 = T5LayerNorm(self.out_ch, layer_norm_eps)

    def forward(self, table, a_simi, t_simi, dep_matrix):  # [B,L,L,dim],[B,L],[B,L],[B,L,L]
        batch, seq, seq, dim = table.shape
        dep_matrix = dep_matrix[:, :seq, :seq]  # 截取句法依存矩阵的有效部分
        dep_matrix = dep_matrix.float()  # 转换为float类型

        # 扩展原有的相似度矩阵
        a_simi2 = a_simi.unsqueeze(2).expand(batch, seq, seq).unsqueeze(3)  # [B,L,L,1]
        t_simi2 = t_simi.unsqueeze(1).expand(batch, seq, seq).unsqueeze(3)  # [B,L,L,1]

        # 处理句法依存矩阵
        dep_weight = dep_matrix.unsqueeze(-1)  # [B,L,L,1]
        dep_weight = self.dep_proj(dep_weight)
        dep_weight = torch.sigmoid(dep_weight)  # 归一化到0-1

        h0 = table  # h_0
        # 第一层传播
        h1_spatial = self.gether(h0, a_simi2, t_simi2)  # 空间依赖
        h1_syntax = self.syntax_gether(h0, dep_weight)  # 句法依赖
        h1 = self.combine_features(h1_spatial, h1_syntax, h0)
        h1 = self.relu(self.norm1(self.dense1(h1)))

        # 第二层传播
        h2_spatial = self.gether(h1, a_simi2, t_simi2)
        h2_syntax = self.syntax_gether(h1, dep_weight)
        h2 = self.combine_features(h2_spatial, h2_syntax, h1)
        h2 = self.relu(self.norm2(self.dense2(h2)))

        return h2

    def gether(self, h, a_simi, t_simi):
        """原有的空间依赖聚合函数"""
        batch, seq, seq, dim = h.shape

        # 水平方向
        padding_y = torch.zeros([batch, seq, 1, dim]).cuda()
        h_y = h * t_simi
        h_left = torch.cat([padding_y, h_y], dim=2)[:, :, :-1, :]
        h_right = torch.cat([h_y, padding_y], dim=2)[:, :, 1:, :]

        # 垂直方向
        padding_x = torch.zeros([batch, 1, seq, dim]).cuda()
        h_x = h * a_simi
        h_up = torch.cat([padding_x, h_x], dim=1)[:, :-1, :, :]
        h_down = torch.cat([h_x, padding_x], dim=1)[:, 1:, :, :]

        h = h + h_left + h_right + h_up + h_down
        return h

    def syntax_gether(self, h, dep_weight):
        """句法依存特征聚合"""
        # 使用句法依存权重调制特征
        h_syntax = h * dep_weight

        # 可选：添加自注意力机制来增强句法依存的表示
        batch, seq, seq, dim = h.shape
        h_syntax = h_syntax + torch.matmul(dep_weight, h_syntax)

        return h_syntax

    def combine_features(self, h_spatial, h_syntax, h_orig):
        """组合空间依赖和句法依存特征"""
        # 计算融合门控值
        gate_input = torch.cat([h_spatial, h_syntax], dim=-1)
        gate = torch.sigmoid(self.dep_gate(gate_input))

        # 动态融合两种特征
        h_combined = gate * h_spatial + (1 - gate) * h_syntax

        # 添加残差连接
        h_final = h_combined + h_orig

        return h_final
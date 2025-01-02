import math
import torch

class SyntaxMultiAttentionGAT(torch.nn.Module):
    def __init__(self, in_channel=768, out_channel=256, num_heads=8, dropout=0.1):
        super(SyntaxMultiAttentionGAT, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.num_heads = num_heads
        self.head_dim = out_channel // num_heads
        assert self.head_dim * num_heads == out_channel, "out_channel must be divisible by num_heads"

        # 多头注意力投影层
        self.q_proj = torch.nn.Linear(in_channel, out_channel)
        self.k_proj = torch.nn.Linear(in_channel, out_channel)
        self.v_proj = torch.nn.Linear(in_channel, out_channel)
        self.o_proj = torch.nn.Linear(out_channel, out_channel)

        # 句法依存投影层
        self.dep_proj = torch.nn.Linear(1, num_heads)

        # 其他层
        self.dropout = torch.nn.Dropout(p=dropout)
        self.layer_norm1 = torch.nn.LayerNorm(out_channel)
        self.layer_norm2 = torch.nn.LayerNorm(out_channel)
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(out_channel, out_channel * 4),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(out_channel * 4, out_channel)
        )

    def forward(self, table, dep_matrix, mask=None):
        """
        Args:
            table: [B, L, L, D] 输入特征
            dep_matrix: [B, L, L] 句法依存矩阵
            mask: [B, L] 可选的掩码（用于忽略padding）
        """
        B, L, L, D = table.shape
        dep_matrix = dep_matrix[:, :L, :L]  # 截断多余部分

        # 1. 多头注意力计算
        # 将输入投影到查询、键、值空间
        q = self.q_proj(table).view(B, L, L, self.num_heads, self.head_dim)
        k = self.k_proj(table).view(B, L, L, self.num_heads, self.head_dim)
        v = self.v_proj(table).view(B, L, L, self.num_heads, self.head_dim)

        # 调整维度顺序为 [B, num_heads, L, L, head_dim]
        q = q.permute(0, 3, 1, 2, 4)
        k = k.permute(0, 3, 1, 2, 4)
        v = v.permute(0, 3, 1, 2, 4)

        # 2. 计算注意力分数
        # [B, num_heads, L, L, L]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 3. 融合句法依存信息
        # 将依存矩阵扩展到多头形式
        dep_weight = self.dep_proj(dep_matrix.unsqueeze(-1))  # [B, L, L, num_heads]
        dep_weight = dep_weight.permute(0, 3, 1, 2)  # [B, num_heads, L, L]
        dep_weight = dep_weight.unsqueeze(3)  # [B, num_heads, L, 1, L]

        # 结合注意力分数和句法信息
        attn_scores = attn_scores + dep_weight

        # 4. 应用掩码（如果提供）
        if mask is not None:
            # 扩展掩码维度
            mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        # 5. Softmax和Dropout
        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # 6. 注意力加权和输出投影
        # [B, num_heads, L, L, head_dim]
        context = torch.matmul(attn_probs, v)
        # [B, L, L, out_channel]
        context = context.permute(0, 2, 3, 1, 4).contiguous()
        context = context.view(B, L, L, self.out_channel)
        output = self.o_proj(context)

        # 7. 残差连接和层归一化
        output = self.layer_norm1(output + table)

        # 8. 前馈网络
        ff_output = self.ffn(output)
        final_output = self.layer_norm2(ff_output + output)

        return final_output

    def _reset_parameters(self):
        """初始化模型参数"""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)
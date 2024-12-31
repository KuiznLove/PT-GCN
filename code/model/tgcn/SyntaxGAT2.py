import torch
from transformers.models.t5.modeling_t5 import T5LayerNorm

class SyntaxGAT2(torch.nn.Module):
    def __init__(self, in_channel=768, out_channel=256, label_num=6):
        super(SyntaxGAT2, self).__init__()
        self.in_ch = in_channel
        self.out_ch = out_channel
        self.label_num = label_num
        self.dropout = torch.nn.Dropout(p=0.5)
        self.relu = torch.nn.ReLU()
        dropout = 0.5
        self.dense1 = torch.nn.Linear(self.in_ch, self.in_ch)
        self.dense2 = torch.nn.Linear(self.in_ch, self.out_ch)

        layer_norm_eps = 1e-12
        self.norm1 = T5LayerNorm(in_channel, layer_norm_eps)
        self.norm2 = T5LayerNorm(self.out_ch, layer_norm_eps)

    def forward(self, table, a_simi, t_simi, adj_matrix):  # adj_matrix: [B, L, L]
        batch, seq, seq, dim = table.shape
        a_simi2 = a_simi.unsqueeze(2).expand(batch, seq, seq).unsqueeze(3)  # [B,L,L,1]
        t_simi2 = t_simi.unsqueeze(1).expand(batch, seq, seq).unsqueeze(3)  # [B,L,L,1]

        h0 = table  # h_0
        h1 = self.gether(h0, a_simi2, t_simi2, adj_matrix)
        h1 = self.relu(self.norm1(self.dense1(h1 + h0)))  # h_1

        h2 = self.gether(h1, a_simi2, t_simi2, adj_matrix)
        h2 = self.relu(self.norm2(self.dense2(h2 + h1)))

        return h2

    def gether(self, h, a_simi, t_simi, adj_matrix):  # adj_matrix: [B, L, L]
        batch, seq, seq, dim = h.shape

        adj_matrix1 = adj_matrix[:, :seq, :seq]

        # 1. 根据句法依存关系聚合邻居信息
        h_adj = torch.einsum('blld,blk->blkd', h, adj_matrix1)  # [B, L, L, dim]

        # 2. 原有的上下左右邻居聚合
        padding_y = torch.zeros([batch, seq, 1, dim]).cuda()  # [B,1,L,dim]
        h_y = h * t_simi
        h_left = torch.cat([padding_y, h_y], dim=2)[:, :, :-1, :]  # [B,L+1,L,dim]
        h_right = torch.cat([h_y, padding_y], dim=2)[:, :, 1:, :]

        padding_x = torch.zeros([batch, 1, seq, dim]).cuda()  # [B,1,dim]
        h_x = h * a_simi
        h_up = torch.cat([padding_x, h_x], dim=1)[:, :-1, :, :]  # [B,L+1,L,dim]
        h_down = torch.cat([h_x, padding_x], dim=1)[:, 1:, :, :]

        # 3. 综合句法依存关系和上下左右邻居信息
        h = h + h_adj + h_left + h_right + h_up + h_down
        return h
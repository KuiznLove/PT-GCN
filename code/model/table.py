import torch
from torch import nn
from .seq2mat import *
from .tgcn.GAT import GAT
from .tgcn.SyntaxGAT import SyntaxGAT
from .tgcn.SyntaxGAT2 import SyntaxGAT2
from .tgcn.SyntaxAttentionGAT import SyntaxAttentionGAT
from .tgcn.SyntaxAttentionGAT2 import SyntaxAttentionGAT2
from .tgcn.GAT import HierarchicalGAT
from .tgcn.SyntaxMultiAttentionGAT import SyntaxMultiAttentionGAT


class TableEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        if config.seq2mat == 'tensor':
            self.seq2mat = TensorSeq2Mat(config)
        elif config.seq2mat == 'tensorcontext':
            self.seq2mat = TensorcontextSeq2Mat(config)
        elif config.seq2mat == 'context':
            self.seq2mat = ContextSeq2Mat(config)
        else:
            self.seq2mat = Seq2Mat(config)


    def forward(self, seq):
        table = self.seq2mat(seq, seq)

        return table

class Ptgcn(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.gat = GAT()
        # self.gat = SyntaxGAT2()
        # self.gat = SyntaxAttentionGAT()
        self.gat = SyntaxAttentionGAT2()
        # self.gat = SyntaxMultiAttentionGAT()
        # self.gat = HierarchicalGAT()

    def forward(self, table, a_s, o_s, adj_matrix):
        table = self.gat(table, a_s, o_s, adj_matrix)
        return table

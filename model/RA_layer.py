import torch.nn as nn
import torch

class RA_Layer(nn.Module):
    def __init__(self, channels):
        super(RA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x,rel_pos_emb):
        x_q = self.q_conv(x).permute(0, 2, 1)# b, n, c
        x_k = self.k_conv(x)# b, c, n
        x_v = self.v_conv(x)

        # b, n, n
        energy = torch.bmm(x_q, x_k)

        # attention = self.softmax(energy)
        attention = self.softmax(energy+rel_pos_emb)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
class RA_Layer1(nn.Module):
    def __init__(self, channels):
        super(RA_Layer1, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)# b, n, c

        x_k = self.k_conv(x)# b, c, n
        x_v = self.v_conv(x)

        # b, n, n
        energy = torch.bmm(x_q, x_k)

        # attention = self.softmax(energy)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))

        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x
'''
Author: chunyangf@qq.com
LastEditors: chunyang fu
Description: Tranformer attention model
Date: 2025-12-06 18:01:45
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from networkTool import device
import copy


class MLPCell(nn.Module):

    def __init__(self, hidden_size, dropout=0, bias=True):
        super(MLPCell, self).__init__()
        layers = []
        for i, num_out_channel in enumerate(hidden_size):
            if (i > 0 and i < len(hidden_size) - 1):
                layers.append(nn.Linear(hidden_size[i - 1], num_out_channel, bias=bias))
                layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_size[-2], hidden_size[-1]))
        layers.append(nn.LayerNorm(num_out_channel, eps=1e-5))
        self.hiddenLayer = nn.Sequential(*layers)

    def forward(self, x):
        out = self.hiddenLayer(x)
        return out


class SelfMultiheadAttention(nn.Module):

    def __init__(self, emsize, nhead, dropout=0, qkv_bias=True):
        super().__init__()
        self.nhead = nhead
        self.head_size = emsize // nhead  # 每个注意力头的维度
        assert emsize % nhead == 0, 'emsize must be times of heads, found {emsize} {nhead}'
        self.all_head_size = int(self.nhead * self.head_size)  # 注意力头拼接起来后的维度
        self.mlpKey = nn.Linear(emsize, self.all_head_size, bias=qkv_bias)
        self.mlpQuery = nn.Linear(emsize, self.all_head_size, bias=qkv_bias)
        self.mlpValue = nn.Linear(emsize, self.all_head_size, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

    # Slice the output of mlpKQV to implement multi-head attention.
    # input:  x.shape = (bs,N,C)  output: x.shape = (bs,nhead,N,C/nhead)
    def slice(self, x):
        new_x_shape = x.size()[:-1] + (self.nhead, self.head_size)  # [batch_size, bptt, nhead, head_size] or [batch_size, bptt, levelNumK, nhead, head_size]
        x = x.view(*new_x_shape)
        x = x.permute(0, 2, 1, 3)
        return x

    # em.shape = [bptt,batch_size,emsize]  mask.shape=[bptt, bptt]
    def forward(self, x, mask):
        x = x.transpose(0, 1).contiguous()  # (bs,bptt,C)
        B_, N, C = x.shape
        Key = self.slice(self.mlpKey(x))  # (bs, bptt, all_head_size) -> (bs,nhead,bptt,head_size)
        Query = self.slice(self.mlpQuery(x))
        Value = self.slice(self.mlpValue(x))
        attention_score = torch.matmul(Query, Key.transpose(-1, -2)) / math.sqrt(self.head_size)  # (bs,nhead,bptt,bptt)

        if mask is not None:
            nW = mask.shape[0]
            # attn = attention_score.view(B_ // nW, nW, self.nhead, N, N) + mask.to(device).unsqueeze(1).unsqueeze(0)
            attn = attention_score + mask
            attn = attn.view(-1, self.nhead, N, N)
            attn = nn.Softmax(dim=-1)(attn)
        else:
            attn = nn.Softmax(dim=-1)(attention_score)

        context = torch.matmul(attn, Value)  # (bs,nhead,bptt,bptt) * (bs,nhead,bptt,head_size)
        context = context.permute(0, 2, 1, 3).contiguous()  # (bs,bptt,nhead,head_size)
        context_shape = context.size()[:-2] + (self.all_head_size, )  # (bs,bptt,all_head_size)
        context = context.view(*context_shape)
        context = context.transpose(0, 1).contiguous()
        return context


class CrossAttention(nn.Module):

    def __init__(self, dim, nhead, qkv_bias=True):
        super().__init__()
        self.nhead = nhead
        self.head_size = dim // self.nhead
        self.mlpK = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlpQ = nn.Linear(dim, dim, bias=qkv_bias)
        self.mlpV = nn.Linear(dim, dim, bias=qkv_bias)
        self.LNK, self.LNQ, self.LNV = nn.LayerNorm(self.head_size, eps=1e-5), nn.LayerNorm(self.head_size, eps=1e-5), nn.LayerNorm(self.head_size, eps=1e-5)

    def forward(self, K, Q, V):
        K, Q, V = K.permute(1, 0, 2), Q.permute(1, 0, 2), V.permute(1, 0, 2)
        bs, K_node_num, C = K.shape
        Q_node_num = Q.shape[1]
        Key = self.LNK(self.mlpK(K).view(bs, K_node_num, self.nhead, self.head_size).permute(0, 2, 1, 3))  # (bs,nhead,num_points,head_dim)
        Query = self.LNQ(self.mlpQ(Q).view(bs, Q_node_num, self.nhead, self.head_size).permute(0, 2, 1, 3))
        Value = self.LNV(self.mlpV(V).view(bs, K_node_num, self.nhead, self.head_size).permute(0, 2, 1, 3))
        attention_score = torch.matmul(Query, Key.transpose(-1, -2)) / math.sqrt(self.head_size)  # (bs,nhead,Q,K)
        attn_map = nn.Softmax(dim=-1)(attention_score)
        x = torch.matmul(attn_map, Value).permute(0, 2, 1, 3).reshape(bs, Q_node_num, C)  # (bs,nhead,Q,head_dim) -> (bs,Q,outdim)
        return x.transpose(0, 1).contiguous()


class TransformerLayer(nn.Module):

    def __init__(self, ninp, nhead, nhid, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.MultiAttention = SelfMultiheadAttention(emsize=ninp, nhead=nhead, dropout=0)
        self.linear1 = nn.Linear(ninp, nhid)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(nhid, ninp)

        self.norm1 = nn.LayerNorm(ninp, eps=1e-5)  # It will affect parallel coding
        self.norm2 = nn.LayerNorm(ninp, eps=1e-5)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    # src is the integration of leaf node and its ancestors.
    def forward(self, src, src_mask):
        src2 = self.MultiAttention(src, src_mask)  # Multi-head Attention
        src = self.dropout1(src2) + src
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))  # [batch_size,bptt,ninp] -> [batch_size,bptt,nhid] -> [batch_size,bptt,ninp]
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class CrossAttnLayer(nn.Module):

    def __init__(self, dim, nhead):
        super().__init__()
        # self.window_size = window_size
        self.cross_attn = CrossAttention(dim=dim, nhead=nhead)
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-5)
        self.norm2 = nn.LayerNorm(dim, eps=1e-5)
        self.actfn = nn.GELU()

    def forward(self, K, Q, V=None):
        L, B, C = K.shape
        shortcut = Q
        # partition windows
        K_windows = K  # window_size, num_windows*B, C
        Q_windows = Q
        V_windows = K
        # cross attention
        attn_windows = self.cross_attn(K=K_windows, Q=Q_windows, V=V_windows)  # window_size, num_windows*B, C
        # merge windows
        x = attn_windows
        # x = merge_windows(attn_windows,window_num=L//self.window_size) # L, B, C
        x = self.norm1(x + shortcut)
        # FFN
        shortcut = x
        x = self.linear2(self.actfn(self.linear1(x)))  # [batch_size,bptt,ninp] -> [batch_size,bptt,nhid] -> [batch_size,bptt,ninp]
        src = self.norm2(x + shortcut)
        return src


class TransformerModule(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0):
        super(TransformerModule, self).__init__()
        self.layers = torch.nn.ModuleList([TransformerLayer(ninp, nhead, nhid, dropout) for _ in range(nlayers)])

    def forward(self, src, src_mask):
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=src_mask)
        return output


class CrossAttnModule(nn.Module):

    def __init__(self, dim, nhead, nlayers):
        super(CrossAttnModule, self).__init__()
        self.layers = torch.nn.ModuleList([CrossAttnLayer(dim, nhead) for _ in range(nlayers)])

    def forward(self, K, Q):
        for mod in self.layers:
            Q = mod(K, Q)
        return Q

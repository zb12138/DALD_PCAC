import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from attentionModel import TransformerModule
from networkTool import ATTIBUTE_RES_RANGE, KNN, device

import shutil
##########################
ninp = 80 + KNN * 12  # embedding dimension
nhid = 300  # the dimension of the feedforward network model in nn.TransformerEncoder
Thxy = [0, 1, 3]
Thz = [0, 1, 3]
LThxy = len(Thxy)
LThz = len(Thz)


class DALD_RGB(nn.Module):

    def __init__(self):
        super(DALD_RGB, self).__init__()
        self.model_type = 'Transformer'
        indiv_dim = 3 + KNN * 33
        self.transformer_encoder = TransformerModule(indiv_dim, 1, nhid, 4)
        self.GEO_encoder0 = nn.Linear(15, 20)
        self.GEO_encoder1 = nn.Linear(20, 28)

        self.nn_encoder = nn.Linear(7, 80)
        self.RGB_encoder0 = nn.Embedding(32, 48)
        self.RGB_encoder1 = nn.Embedding(32, 48)
        self.RGB_encoder2 = nn.Embedding(32, 48)

        self.ATR_encoder = nn.Linear(144, 48)
        self.ninp = ninp
        self.act = nn.ReLU()

        group_dim = 3
        self.decoder00 = nn.Linear(indiv_dim + group_dim, ninp)
        self.decoder10 = nn.Linear(ninp, ATTIBUTE_RES_RANGE)

        self.decoderR2G = nn.Linear(indiv_dim + group_dim + 1, indiv_dim + group_dim)

        self.decoder01 = nn.Linear(indiv_dim + group_dim, ninp)
        self.decoder11 = nn.Linear(ninp, ATTIBUTE_RES_RANGE)

        self.decoderRG2B = nn.Linear(indiv_dim + group_dim + 2, indiv_dim + group_dim)

        self.decoder02 = nn.Linear(indiv_dim + group_dim, ninp)
        self.decoder12 = nn.Linear(ninp, ATTIBUTE_RES_RANGE)

        self.Embdl = nn.Embedding((2 * LThxy + 1) * (2 * LThxy + 1) * (2 * LThz + 1), 6)
        self.EmbdrR = nn.Embedding(511, 6)
        self.EmbdrG = nn.Embedding(1023, 6)
        self.EmbdrB = nn.Embedding(1023, 6)

        self.EmbdaR = nn.Embedding(511, 3)
        self.EmbdaG = nn.Embedding(1023, 3)
        self.EmbdaB = nn.Embedding(1023, 3)

        self.init_weights()

    def Embda(self, x):
        return torch.cat((self.EmbdaR(x[..., 0:1]), self.EmbdaG(x[..., 1:2]), self.EmbdaB(x[..., 2:3])), -1)

    def Embdr(self, x):
        return torch.cat((self.EmbdrR(x[..., 0:1]), self.EmbdrG(x[..., 1:2]), self.EmbdrB(x[..., 2:3])), -1)

    def init_weights(self):

        def init_w(xL, bias=False):
            for x in xL:
                if bias:
                    x.bias.data.zero_()
                x.weight.data = nn.init.xavier_normal_(x.weight.data)

        init_w([self.GEO_encoder0, self.nn_encoder, self.ATR_encoder], bias=True)
        init_w([self.RGB_encoder0, self.RGB_encoder1, self.RGB_encoder2])
        init_w([self.decoder00, self.decoder01, self.decoder02], bias=True)
        init_w([self.decoder10, self.decoder11, self.decoder12], bias=True)
        init_w([self.decoderR2G, self.decoderRG2B], bias=True)

    def xyz_quan(self, x, quan_in):
        ax = torch.abs(x)
        quan_result = torch.zeros_like(x)
        quan_inter = [q for q in quan_in + [ax.max() + 1]]
        for i, q in enumerate(quan_inter):
            quan_result += (torch.sign(x) * (ax < q) * (ax >= quan_inter[i - 1])) * i
        return quan_result.long()

    def forward(self, data, targets):
        output = self.forward_base(data, targets)
        outputr = self.forward1(output, targets)
        outputg = self.forward2(output, targets)
        outputb = self.forward3(output, targets)
        return torch.cat([outputr[None, :], outputg[None, :], outputb[None, :]], 0).permute(1, 2, 0, 3)

    def forward_base(self, data, targets):
        pi = data['geo'][:, :, 0:3].unsqueeze(2).to(device).float()
        pj = data['nn'][:, :, :, 0:3].to(device).float()
        aj = data['nn'][:, :, :, 3:6].to(device).long()
        ahati = data['predint'][:, :, 0:3].to(device).unsqueeze(2).long()  # ahati
        rj = data['resnn'][:, :, :, 0:3].to(device).long() + torch.Tensor([255, 511, 511]).cuda().long()

        xyz = pj - pi
        xy_q = self.xyz_quan(xyz[..., :2], Thxy) + LThxy
        z_q = self.xyz_quan(xyz[..., 2:], Thz) + LThz
        L = self.Embdl(xy_q[..., 0] + xy_q[..., 1] * (2 * LThxy + 1) + z_q[..., 0] * (2 * LThxy + 1) * (2 * LThxy + 1))

        # geo norm
        shift = pi.min(1, keepdim=True)[0]
        pi -= shift
        pj -= shift
        minmax = pi.max(1, keepdim=True)[0]
        minmax[minmax == 0] = 1
        pi /= minmax
        pj /= minmax

        # construct graph # scipy.spatial.KDTree.sparse_distance_matrix
        batch = pi.shape[0]
        vertex = pi.shape[1]

        # eaj = self.Embda(aj).squeeze(-2)
        ri = aj - ahati + torch.Tensor([255, 511, 511]).cuda().long()
        # assert ri.min()>=0 and ri.max()<=1022,data['file']
        # assert ahati.min()>=0 and ahati.max()<=511,data['file']
        eri = self.Embdr(ri).squeeze(-2)
        eaj = self.Embda(rj).squeeze(-2)
        Fj = torch.cat((L, eri, eaj), -1)
        Fi = pi
        graph = torch.cat((Fi.reshape(batch, vertex, -1), Fj.reshape(batch, vertex, -1)), -1)  # torch.Size([4, 1024, 10, 6])

        transformerInput = graph.reshape(batch, vertex, -1).permute(1, 0, 2)
        output_cross = self.transformer_encoder(transformerInput, src_mask=None).permute(1, 0, 2)
        # output_cross = transformerInput.permute(1,0,2)
        base_context = torch.cat((output_cross, aj.float().std(2)), -1)
        return base_context

    def forward1(self, base_context, targets):
        outputr = self.decoder10(self.act(self.decoder00(base_context)))
        return outputr

    def forward2(self, base_context, targets):
        outputr2g = self.act(self.decoderR2G(torch.cat((targets[:, :, 0:1], base_context), -1)))
        outputg = self.decoder11(self.act(self.decoder01(outputr2g)))
        return outputg

    def forward3(self, base_context, targets):
        outputrg2b = self.act(self.decoderRG2B(torch.cat((targets[:, :, 0:2], base_context), -1)))
        outputb = self.decoder12(self.act(self.decoder02(outputrg2b)))
        return outputb


class DALD_LiDAR(nn.Module):

    def __init__(self):
        super(DALD_LiDAR, self).__init__()
        self.model_type = 'Transformer'
        indiv_dim = 6 + KNN * 21
        self.transformer_encoder = TransformerModule(indiv_dim, 1, nhid, 4)
        self.GEO_encoder0 = nn.Linear(15, 20)
        self.GEO_encoder1 = nn.Linear(20, 28)

        self.nn_encoder = nn.Linear(7, 80)
        self.RGB_encoder0 = nn.Embedding(32, 48)
        self.RGB_encoder1 = nn.Embedding(32, 48)
        self.RGB_encoder2 = nn.Embedding(32, 48)

        self.ATR_encoder = nn.Linear(144, 48)
        self.ninp = ninp
        self.act = nn.ReLU()

        group_dim = 1
        self.decoder00 = nn.Linear(indiv_dim + group_dim, ninp)
        self.decoder10 = nn.Linear(ninp, ATTIBUTE_RES_RANGE)

        self.decoder01 = nn.Linear(indiv_dim + group_dim, ninp)
        self.decoder11 = nn.Linear(ninp, 4)

        self.decoder02 = nn.Linear(indiv_dim + group_dim, ninp)
        self.decoder12 = nn.Linear(ninp, 32)

        self.decoder03 = nn.Linear(80, ninp)
        self.decoder13 = nn.Linear(ninp, 12)

        self.Embdl = nn.Embedding(343, 6)
        self.Embdr = nn.Embedding(511, 6)
        self.Embda = nn.Embedding(255, 3)

        self.init_weights()

    def init_weights(self):

        def init_w(xL, bias=False):
            for x in xL:
                if bias: x.bias.data.zero_()
                x.weight.data = nn.init.xavier_normal_(x.weight.data)

        init_w([self.GEO_encoder0, self.nn_encoder, self.ATR_encoder], bias=True)
        init_w([self.RGB_encoder0, self.RGB_encoder1, self.RGB_encoder2])
        init_w([self.decoder00, self.decoder01, self.decoder02, self.decoder03], bias=True)
        init_w([self.decoder10, self.decoder11, self.decoder12, self.decoder13], bias=True)

    def xyz_quan(self, x, quan_in):
        ax = torch.abs(x)
        s = ax.reshape(ax.shape[0], -1, ax.shape[-1]).mean(1, keepdim=True).unsqueeze(1)
        quan_result = torch.zeros_like(x)
        quan_inter = [s * q for q in quan_in + [ax.max() + 1]]
        for i, q in enumerate(quan_inter):
            quan_result += (torch.sign(x) * (ax < q) * (ax >= quan_inter[i - 1])) * i
        return quan_result.long()


    def forward(self, data, targets):
        pi = data['geo'][:, :, 0:3].unsqueeze(2).to(device).float()
        pj = data['nn'][:, :, :, 0:3].to(device).float()
        ahati = data['predint'][:, :, 0:1].to(device).unsqueeze(2).long()  # ahati
        aj = data['nn'][:, :, :, 3:4].to(device).long()

        xyz = pj - pi
        xy_q = self.xyz_quan(xyz[..., :2], Thxy) + LThxy
        z_q = self.xyz_quan(xyz[..., 2:], Thz) + LThz
        L = self.Embdl(xy_q[..., 0] + xy_q[..., 1] * (2 * LThxy + 1) + z_q[..., 0] * (2 * LThxy + 1) * (2 * LThxy + 1))

        # geo norm
        shift = pi.min(1, keepdim=True)[0]
        pi -= shift
        pj -= shift
        minmax = pi.max(1, keepdim=True)[0]
        minmax[minmax == 0] = 1
        pi /= minmax
        pj /= minmax

        # construct graph # scipy.spatial.KDTree.sparse_distance_matrix

        batch = pi.shape[0]
        vertex = pi.shape[1]

        eai = self.Embda(ahati).squeeze(-2)
        eaj = self.Embda(aj).squeeze(-2)
        eri = self.Embdr(aj - ahati + ATTIBUTE_RES_RANGE // 2).squeeze(-2)

        Fj = torch.cat((pj, pj - pi, eaj, eri, L), -1)
        Fi = torch.cat((pi, eai), -1)
        graph = torch.cat((Fj.reshape(batch, vertex, -1), Fi.reshape(batch, vertex, -1)), -1)  # torch.Size([4, 1024, 10, 6])

        transformerInput = graph.reshape(batch, vertex, -1).permute(1, 0, 2)
        output_cross = self.transformer_encoder(transformerInput, src_mask=None).permute(1, 0, 2)
        output = torch.cat((output_cross, aj.float().std(2)), -1)

        outputr = self.decoder10(self.act(self.decoder00(output)))
        return outputr

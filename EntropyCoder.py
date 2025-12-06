'''
Author: chunyangf@qq.com
LastEditors: chunyang fu
Description: Entropy coder file
Date: 2025-12-06 16:20:33
'''
# entropy encoder
from resAc.ac_warpper import encode_res, decode_res
import torch
import torchac
import os
import glob
import numpy as np
import testTool.ptIO as ptIO
import testTool.ptfun as ptfun
from testTool.testTMC import Cpt
from networkTool import EXPNAME
import numpyAc.GPU_AC as GPU_AC


class TEntropyCoder():

    def __init__(self, bin_path="", actual_code=True) -> None:
        self.bin_path = bin_path
        self.actual_code = actual_code
        self.ac_coder = 'GPU_AC_GPU'  # 'GPU_AC_GPU'

    def gpcc_encode(self, pt, colorSpase):
        if colorSpase == "YUV" or colorSpase == "rgb":
            if colorSpase == "YUV":
                pt = np.c_[pt[:, :3], ptfun.YCoCg2RGB(pt[:, 3:6])]
            ptIO.write_ply_data(EXPNAME + '/tmp/dense.ply', pt, attributeName=['red', 'green', 'blue'], attriType=['uchar', 'uchar', 'uchar'])
            gpcc = Cpt(EXPNAME + '/tmp/dense.ply', attributeType='rgb')
            gpcc_result = gpcc.compressByTmc(Config='testTool/gpcc_cw.cfg', print_scree=False, BinPath=self.bin_path + 'gpcc')
        elif colorSpase == "ref":
            pt = pt[:, :4].copy()
            ptIO.write_ply_data(EXPNAME + '/tmp/lidar.ply', pt, attributeName=['reflectance'], attriType=['uint16'])
            gpcc = Cpt(EXPNAME + '/tmp/lidar.ply', attributeType='ref')
            gpcc_result = gpcc.compressByTmc(Config='testTool/gpcc_cw_lidar.cfg', print_scree=False, BinPath=self.bin_path + 'gpcc')
        os.system('mv {} {}'.format(self.bin_path + 'gpcc', self.bin_path + f'{int(gpcc_result["atr_bits"])}gpcc'))
        return gpcc_result

    def gpcc_decode(self, pos, colorSpase):
        pos = pos[:, :3]
        attriType = 'ref'
        if colorSpase != 'ref': attriType = 'rgb'
        gpcc = Cpt(path=EXPNAME + '/tmp/base_layer.ply', attributeType=attriType, create_new=True)
        gpcc.deCompressByTmc(BinPath=glob.glob(self.bin_path + '*gpcc')[0], OutputPath=gpcc.path, print_scree=False)
        geo, color = gpcc.read()
        _, midx = ptfun.sortByMorton(pos, return_idx=True)
        if colorSpase == "YUV":
            color = ptfun.RGB2YCoCg(color)
        decolor = np.zeros_like(pos)
        if colorSpase != "ref":
            decolor[midx] = ptfun.sortByMorton(np.c_[geo, color])[:, 3:6]
        else:
            decolor[midx] = np.tile(ptfun.sortByMorton(np.c_[geo, color].astype(int))[:, 3:4], (1, 3))
        return np.c_[pos, decolor]

    def res_encoder(self, data, bintype):
        if len(data) == 0:
            return 0
        code = encode_res(data)
        if bintype and self.actual_code:
            self.write_bin(code, self.bin_path + bintype)
        return len(code) * 8

    def res_decoder(self, bintype, pointNum, channel):
        code = self.read_bin(self.bin_path, bintype)
        return decode_res(code, pointNum, channel)

    def pdf_convert_to_cdf_and_normalize(self, pdf):
        assert pdf.ndim == 2
        cdfF = torch.cumsum(pdf, axis=1)
        cdfF = cdfF / cdfF[:, -1:]
        cdfF = torch.hstack((torch.zeros((pdf.shape[0], 1)), cdfF))
        return cdfF

    def ac_encoder(self, targets, pros, bintype):
        if self.ac_coder == 'GPU_AC_GPU':
            GPU_AC.encode(targets, pros, self.bin_path + bintype, useGpu=True)
        elif self.ac_coder == 'GPU_AC_CPU':
            GPU_AC.encode(targets, pros, self.bin_path + bintype, useGpu=False)
        else:
            pros = self.pdf_convert_to_cdf_and_normalize(pros.cpu())
            code = torchac.encode_float_cdf(pros, targets.cpu().short(), check_input_bounds=True)
            self.write_bin(code, self.bin_path + bintype)
        return os.stat(self.bin_path + bintype).st_size * 8

    def encoder_ref_and_color(self, residuals, pros, slice_id):
        if residuals.shape[1] == 1:
            ref_len = self.ac_encoder(residuals[:, 0], pros[:, 0], 'Rbin{}'.format(slice_id))
            return ref_len, np.array([ref_len, ref_len, ref_len])
        else:
            return self.ac_encoder3(residuals, pros, slice_id)

    def ac_encoder3(self, residuals, pros, bintype):
        residuals = residuals  # .cpu()
        pros = pros  # .cpu()
        codelen = 0
        codelenR = self.ac_encoder(residuals[:, 0], pros[:, 0], 'Rbin{}'.format(bintype))
        codelenG = self.ac_encoder(residuals[:, 1], pros[:, 1], 'Gbin{}'.format(bintype))
        codelenB = self.ac_encoder(residuals[:, 2], pros[:, 2], 'Bbin{}'.format(bintype))
        codelen = codelenR + codelenG + codelenB
        return codelen, np.array([codelenR, codelenG, codelenB])

    def ac_decoder(self, pros, bintype):
        if self.ac_coder == 'GPU_AC_GPU':
            return GPU_AC.decode(pros, self.bin_path + bintype, useGpu=True)
        elif self.ac_coder == 'GPU_AC_CPU':
            return GPU_AC.decode(pros, self.bin_path + bintype, useGpu=False)
        else:
            code = self.read_bin(self.bin_path, bintype)
            return torchac.decode_float_cdf(self.pdf_convert_to_cdf_and_normalize(pros.cpu()), bytes(code))

    def ac_decoder3(self, pros, bintype):
        de_resR = self.ac_decoder(pros[:, 0], 'Rbin{}'.format(bintype)).reshape(-1, 1)
        de_resG = self.ac_decoder(pros[:, 1], 'Gbin{}'.format(bintype)).reshape(-1, 1)
        de_resB = self.ac_decoder(pros[:, 2], 'Bbin{}'.format(bintype)).reshape(-1, 1)
        return torch.cat((de_resR, de_resG, de_resB), -1).cpu().numpy()

    def write_bin(self, code, bitstream_path):
        os.makedirs(os.path.dirname(bitstream_path), exist_ok=True)
        with open(bitstream_path, 'wb') as fout:
            if type(code) is np.ndarray:
                code = code.tolist()
            fout.write(bytes(code))

    def read_bin(self, bitstream_path, bintype=''):
        with open(bitstream_path + bintype, 'rb') as fin:
            bitstream = fin.read()
        return list(bitstream)

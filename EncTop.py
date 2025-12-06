'''
Author: chunyangf@qq.com
LastEditors: chunyang fu
Description: DALD-PCAC encoder file
Date: 2025-12-06 18:02:40
'''
from networkTool import MAX_SLICE_NUM, ATTIBUTE_RES_RANGE, BPTT
from Common import TContext, TPreprocess
import time
import os
from copy import deepcopy
from testTool.testTMC import TestFile
import torch
from collections import defaultdict
import numpy as np
from networkTool import EXPNAME, reload, CKPT_PATH, IS_LIDAR, CPrintl, ATTR_NAME
from EntropyCoder import TEntropyCoder
from train import Train_and_Test


class TEncCf(TPreprocess):
    from train import Model
    model_path = CKPT_PATH
    saveDic = reload(None, model_path)
    assert saveDic is not None
    Model.load_state_dict(saveDic['encoder'])

    def __init__(self) -> None:
        super().__init__()  # TPreprocess
        self.m_inputFileName = ""
        self.m_bitstreamFolderName = ""
        self.m_reconFileName = ""

        self.m_encBac = TEntropyCoder()
        self.m_deepEntroyModel = Train_and_Test(model=TEncCf.Model, model_path=None)  # do not load again
        self.m_deepEntroyModel.model_path = TEncCf.model_path

    def pharseInput(self, inputFileName, out_bin_dir=EXPNAME):
        self.m_inputFileName = inputFileName
        self.m_bitstreamFolderName = os.path.join(out_bin_dir, 'bin', os.path.basename(inputFileName)[:-4])
        self.m_reconFileName = os.path.join(out_bin_dir + '/dec', inputFileName[:-4] + '_rec.ply')

        # for test
        self.m_deepEntroyModel.net_acbit = 0
        self.m_deepEntroyModel.net_celoss = 0
        self.m_deepEntroyModel.net_ptNum = 0
        self.m_deepEntroyModel.net_elapsed = 0


class TEncTop(TEncCf):

    def __init__(self) -> None:
        super().__init__()
        self.m_pointCloudOrg = None
        self.m_pointCloudQuant = None
        self.m_pointCloudRecon = None
        self.m_attrEncoder = None
        self.encode_bitnum = defaultdict(int)
        self.encode_ptnum = 0
        self.actual_code = True
        self.encode_context = None
        self.max_pt_num_in_slice = 10000000

    def encode(self, inputFileName, out_bin_dir=EXPNAME):
        self.__init__()
        self.pharseInput(inputFileName, out_bin_dir)
        self.readOriPointCloud(self.m_inputFileName)
        self.m_pointCloudQuant = self.preprocess(copy_ori_pt=True)
        if os.path.exists(self.m_bitstreamFolderName):
            os.system('rm -rf {}'.format(self.m_bitstreamFolderName))
        resultBk = []
        for i, m_pointCloudSlice in enumerate(self.getSlice(self.m_pointCloudQuant, self.max_pt_num_in_slice)):
            # geo coding:
            self.m_pointCloudRecon = deepcopy(m_pointCloudSlice)
            # atr coding:
            self.m_encBac.bin_path = os.path.join(self.m_bitstreamFolderName, f'bk{i:03d}.')
            result = self.predictAndEncodeAttribute()
            resultBk.append(result)
        return self.sumResult(resultBk)

    def sumResult(self, resultList):
        if len(resultList) == 1:
            resultList[0]['bkNum'] = 1
            return resultList[0]
        result = {}
        sumKey = ['inPtNum', 'outPtNum', 'ConTime', 'NetTime', 'Ac_time', 'EnToTaltime', 'Bits', 'RGB_base']
        for key in sumKey:
            result[key] = sum([x[key] for x in resultList])
        result['bpop'] = result['Bits'] / result['outPtNum']
        result['bkNum'] = len(resultList)
        return result

    def predictAndEncodeAttribute(self):
        self.encode_bitnum = defaultdict(int)
        self.encode_ptnum = 0
        t0 = time.time()
        context = TContext(self.m_pointCloudRecon)
        self.encode_context = context
        base_layer = context.Base_LOD()

        gpcc_codec = self.m_encBac.gpcc_encode(base_layer, self.m_colorSpace)
        self.encode_bitnum['gpcc'] += gpcc_codec['atr_bits']
        self.encode_ptnum += base_layer.shape[0]

        context.Infer_LOD(bptt=BPTT)
        context.Predict()
        context_time = (time.time() - t0)
        attribute_dim = 1 if self.m_colorSpace == 'ref' else 3
        # net

        net_bin_code_len = 0
        net_bin_rgb_len = 0
        net_elapsed = 0
        entropy_coding_time = 0
        actual_code_ptNum = 0
        for slice_id in range(MAX_SLICE_NUM):
            pros, residuals = [], []
            for dict_data in context.construct_context(slice_id):
                for k in ['geo', 'resnn', 'nn', 'predint']:
                    dict_data[k] = torch.IntTensor(dict_data[k]).cuda()
                dict_data['target'] = torch.LongTensor(dict_data['target']).cuda()
                is_not_padding = (dict_data['padding'] == -1)
                t_net = time.time()
                pro = self.m_deepEntroyModel.network(dict_data, test_type='encode')
                net_elapsed += (time.time() - t_net)
                pros.append(pro[is_not_padding, :, :])
                residuals.append(dict_data["target"][is_not_padding, :, :])

            # entropy coding
            residuals = torch.cat(residuals).reshape(-1, attribute_dim)
            actual_code_ptNum += len(residuals)
            t_en = time.time()
            if self.actual_code:
                pros = torch.cat(pros).reshape(-1, attribute_dim, ATTIBUTE_RES_RANGE)
                code_len, rgb_len = self.m_encBac.encoder_ref_and_color(residuals, pros, slice_id)
                net_bin_code_len += code_len
                net_bin_rgb_len += rgb_len
            else:
                net_bin_code_len = self.m_deepEntroyModel.net_acbit
            entropy_coding_time += (time.time() - t_en)

        net_bin_code_len += self.m_encBac.res_encoder(context.out_target, 'outTarget')

        self.encode_bitnum['net'] = net_bin_code_len

        self.encode_ptnum += actual_code_ptNum
        total_time = (time.time() - t0)

        # stat
        assert self.encode_ptnum == self.m_pointCloudRecon.m_numPoint
        bpp = [(a, np.round(b / self.encode_ptnum, 4)) for a, b in self.encode_bitnum.items()]
        bits = sum(self.encode_bitnum.values())
        bpip = bits / self.m_pointCloudRecon.m_numPoint
        result = {
            'File': self.m_pointCloudRecon.m_name,
            'inPtNum': self.m_pointCloudRecon.m_numPoint,
            'outPtNum': self.encode_ptnum,
            'bpop':  bpip,
            'ConTime': context_time,
            'NetTime': net_elapsed, \
            # {net_elapsed,self.m_deepEntroyModel.net_elapsed},\
            'Ac_time': entropy_coding_time, \
            'EnToTaltime': total_time, \
            'Bppinfo': bpp, \
            'Bits': bits, \
            'RGB_base': np.r_[net_bin_rgb_len , self.encode_bitnum['gpcc']]
        }
        return result


if __name__ == '__main__':
    printl = CPrintl(EXPNAME + '/test.log')
    encoder = TEncTop()
    model_path = encoder.m_deepEntroyModel.model_path
    printl('deep entropy model:', model_path)
    if IS_LIDAR:
        test = TestFile(path='Data/MPEG/MPEGCat3Frame/Ford_02_q1mm/*10.ply', attributeType=ATTR_NAME)
    else:
        test = TestFile(path='Dataset/8iVFBv2/*.ply', attributeType=ATTR_NAME)

    result = test.testByFun(encoding_fun=lambda x: encoder.encode(x), print=printl, lidar=IS_LIDAR)
    result.iloc[-1, 0] = model_path
    result.to_csv(EXPNAME + '/EncTop.csv')
    printl(result)
    print('Test finished. Results are saved to {}'.format(EXPNAME + '/test.csv'))

import os
import time
import torch
import numpy as np
from networkTool import MAX_SLICE_NUM, ATTIBUTE_HALF_RANGE, ATTIBUTE_RES_RANGE, CPrintl, EXPNAME, ATTR_NAME, IS_LIDAR
from Common import TContext, TComPointCloud
from EncTop import TEncTop
from testTool.testTMC import run_cmd, TestFile



class TDecTop(TEncTop):

    def __init__(self) -> None:
        super().__init__()
        self.m_pointCloudDecodeGeo = TComPointCloud()
        self.m_pointCloudDecode = TComPointCloud()
        self.m_pointCloudDecode.m_attributeType = ATTR_NAME

    def decodeChannel(self, forwordFun, dict_data, pros: list):
        is_not_padding = dict_data['padding']
        t = time.time()
        output_pro = forwordFun(dict_data['base_context'], dict_data['target'])
        softmaxR = torch.softmax(output_pro.detach(), dim=-1)
        net_elapsed = (time.time() - t)
        pros.append(softmaxR[is_not_padding, :])
        return net_elapsed

    def decodeSlice(self, m_pointCloud_geo):
        t0 = time.time()
        context = TContext(m_pointCloud_geo)
        base_layer = context.Base_LOD()
        recon_base_layer = self.m_encBac.gpcc_decode(base_layer, self.m_colorSpace)
        # assert (recon_base_layer==self.encode_context.base_layer).all()
        context.Update_baselayer(recon_base_layer)
        context.Infer_LOD()
        context.Predict()

        outTarget = self.m_encBac.res_decoder('outTarget', context.p_m.shape[0], 3)

        net_elapsed = 0

        if self.m_attributeType == "rgb":

            # decode net
            for slice_id in range(MAX_SLICE_NUM):
                dict_data_list, decdidx, paddings = [], [], []
                proR, proG, proB = [], [], []
                target_all = torch.zeros((context.target.shape)).short().cuda()

                for dict_data in context.construct_context(slice_id):
                    for k in ['geo', 'resnn', 'nn', 'predint']:
                        dict_data[k] = torch.IntTensor(dict_data[k]).cuda()
                    dict_data['target'] = torch.LongTensor(dict_data['target']).cuda()

                    pointNum = dict_data['geo'].shape[0] * \
                        dict_data['geo'].shape[1]
                    self.m_deepEntroyModel.net_ptNum += pointNum
                    is_not_padding = (dict_data['padding'] == -1)
                    decdidx.append(dict_data['idx'][is_not_padding])
                    paddings.append(np.c_[dict_data['idx'][:, :, None], dict_data['padding'][:, :, None]][~is_not_padding])
                    net_t = time.time()
                    base_context = self.m_deepEntroyModel.model.forward_base(dict_data, None)
                    net_elapsed += (time.time() - net_t)
                    dict_context = {}
                    dict_context['target'] = dict_data['target']
                    dict_context['padding'] = is_not_padding
                    dict_context['base_context'] = base_context
                    dict_context['idx'] = dict_data['idx']
                    dict_data_list.append(dict_context)

                decode_idx = np.concatenate(decdidx).reshape(-1)
                padding_idx = np.concatenate(paddings).reshape(-1, 2)
                # _________________________________________________
                for dict_data in dict_data_list:
                    net_elapsed += self.decodeChannel(self.m_deepEntroyModel.model.forward1, dict_data, proR)
                decode_prosR = torch.cat(proR).reshape(-1, ATTIBUTE_RES_RANGE)
                del proR
                targetR = self.m_encBac.ac_decoder(decode_prosR, 'Rbin{}'.format(slice_id))
                target_all[decode_idx, 0] = targetR.reshape(-1)
                # assert ((self.encode_context.target==target_all.cpu().numpy())[decode_idx,0]).all()
                del decode_prosR
                for dict_data in dict_data_list:
                    dict_data['target'][:, :, 0] = target_all[dict_data['idx'], 0]
                    net_elapsed += self.decodeChannel(self.m_deepEntroyModel.model.forward2, dict_data, proG)
                decode_prosG = torch.cat(proG).reshape(-1, ATTIBUTE_RES_RANGE)
                targetG = self.m_encBac.ac_decoder(decode_prosG, 'Gbin{}'.format(slice_id))
                target_all[decode_idx, 1] = targetG.reshape(-1)
                # assert ((self.encode_context.target==target_all.cpu().numpy())[decode_idx,1]).all()
                del decode_prosG
                del proG
                for dict_data in dict_data_list:
                    dict_data['target'][:, :, 1] = target_all[dict_data['idx'], 1]
                    net_elapsed += self.decodeChannel(self.m_deepEntroyModel.model.forward3, dict_data, proB)
                decode_prosB = torch.cat(proB).reshape(-1, ATTIBUTE_RES_RANGE)
                targetB = self.m_encBac.ac_decoder(decode_prosB, 'Bbin{}'.format(slice_id))
                target_all[decode_idx, 2] = targetB.reshape(-1)
                # assert ((self.encode_context.target==target_all.cpu().numpy())[decode_idx,2]).all()
                del decode_prosB
                del proB
                # _________________________________________________
                context.p_m[decode_idx, 3:6] = context.pred_yuv[decode_idx] + \
                    target_all[decode_idx].cpu().numpy() - \
                    ATTIBUTE_HALF_RANGE + outTarget[decode_idx]
                context.p_m[padding_idx[:, 0], 3:6] = context.p_m[padding_idx[:, 1], 3:6]  # padding
                # assert ( self.encode_context.p_m[decode_idx,:6] == context.p_m[decode_idx,:6] ).all()
                # context.Update_pred(decode_idx)
                # only need update next slice
                context.Update_pred(np.where((context.p_m[:, 7] == (-slice_id - 1)))[0])

            m_color = context.p_m[context.p_m[:, 8] == -1, 3:6]
            m_pos = context.p_m[context.p_m[:, 8] == -1, 0:3]

        if self.m_attributeType == 'ref':
            # decode net
            for slice_id in range(MAX_SLICE_NUM):
                pros, decdidx = [], []
                for dict_data in context.construct_context(slice_id):
                    for k in ['geo', 'resnn', 'nn', 'predint']:
                        dict_data[k] = torch.IntTensor(dict_data[k]).cuda()
                    dict_data['target'] = torch.LongTensor(dict_data['target']).cuda()
                    is_not_padding = (dict_data['padding'] == -1)
                    t = time.time()
                    pro = self.m_deepEntroyModel.network(dict_data, test_type='decode')
                    decdidx.append(dict_data['idx'][is_not_padding])
                    net_elapsed += (time.time() - t)
                    pros.append(pro[is_not_padding, :, :])

                pros = torch.cat(pros).reshape(-1, 1, ATTIBUTE_RES_RANGE)
                decode_idx = np.concatenate(decdidx).reshape(-1)
                target_all = self.m_encBac.ac_decoder(pros[:, 0], 'Rbin{}'.format(slice_id)).reshape(-1)
                context.p_m[decode_idx, 3:6] = context.pred_yuv[decode_idx] + np.tile(target_all.cpu().numpy(), (3, 1)).T - ATTIBUTE_HALF_RANGE + outTarget[decode_idx]
                context.Update_pred()

            m_color = context.p_m[:, 3:6]
            m_pos = context.p_m[:, 0:3]

        self.m_pointCloudDecode.addPoints(m_pos, m_color)
        total_time = time.time() - t0
        # run_cmd('testTool/pc_error -a {} -b {} -c 1'.format(self.m_inputFileName,self.m_reconFileName),True)
        return {'DeNet_time': net_elapsed, 'totalDetime': total_time}

    def decode(self, inputFileName, bin_dir=EXPNAME, decodeFileName=None):
        # encodeRuslt = self.encode(inputFileName)
        self.__init__()
        self.pharseInput(inputFileName, bin_dir)
        self.m_encBac.bin_path = self.m_encBac.bin_path = os.path.join(self.m_bitstreamFolderName, f'bk{0:03d}.')
        if decodeFileName is not None:
            self.m_reconFileName = decodeFileName
        # dec geo:
        self.m_pointCloudDecodeGeo.readFromFile(inputFileName, atriType="xyz")
        self.m_pointCloudDecodeGeo.quantization(self.m_quanParm)

        # set dummy color
        self.m_pointCloudDecodeGeo.clearColors()
        net_elapsed = 0
        total_time = 0
        bits = 0
        try:
            _,_,binfile = list(os.walk(self.m_bitstreamFolderName))[0]
        except:
            assert False, f"bitstream folder {self.m_bitstreamFolderName} not found!"
        for f in binfile:
            if f.endswith('gpcc'):
                bits += int(f.split('.')[1].split('gpcc')[0])
                continue
            bits += (os.path.getsize(os.path.join(self.m_bitstreamFolderName,f))*8)
        for i, m_pointCloudSlice in enumerate(self.getSlice(self.m_pointCloudDecodeGeo, self.max_pt_num_in_slice)):
            m_pointCloudSlice.m_attributeType = self.m_pointCloudDecode.m_attributeType
            self.m_encBac.bin_path = os.path.join(self.m_bitstreamFolderName, f'bk{i:03d}.')
            result = self.decodeSlice(m_pointCloudSlice)
            net_elapsed += result['DeNet_time']
            total_time += result['totalDetime']
        self.m_pointCloudDecode.deQuantization(self.m_quanParm)
        if self.m_colorSpace == "YUV":
            self.m_pointCloudDecode.convertYUVToRGB()
        self.m_pointCloudDecode.saveToFile(self.m_reconFileName,asAscii=False)
        return {'DeNet_time': net_elapsed, 'totalDetime': total_time, 'inPtNum': self.m_pointCloudDecode.m_numPoint, 'Bits': bits}


if __name__ == '__main__':
    import os
    printl = CPrintl(EXPNAME + '/test.log')
    encoder = TEncTop()
    decoder = TDecTop()
    if IS_LIDAR:
        test = TestFile(path='Data/MPEG/MPEGCat3Frame/Ford_02_q1mm/*10.ply',attributeType=ATTR_NAME)
    else:
        test = TestFile(path='Dataset/8iVFBv2/*.ply',attributeType=ATTR_NAME)
    result = test.testByFun(encoding_fun=lambda x: dict(**(decoder.decode(x))), print=printl, testGPCC=['enc','dec'], lidar=IS_LIDAR)
    result.iloc[-1, 0] = encoder.m_deepEntroyModel.model_path
    result.to_csv(EXPNAME + '/DecTop.csv')
    printl(result)

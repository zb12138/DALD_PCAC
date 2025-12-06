'''
Author: chunyangf@qq.com
LastEditors: chunyang fu
Description: Common coder framework
Date: 2025-12-06 18:01:18
'''
import numpy as np
import testTool.ptIO as ptIO
from testTool.ptfun import kdtree_partition, YCoCg2RGB, RGB2YCoCg
from LOD import pred_from_lod
from networkTool import KNN, INTRA_LOD_START, ATTIBUTE_HALF_RANGE, MAX_DIS, ATTR_NAME
import numpy as np
import os
from LOD import levelOfDetailLayeringStructure, InferLodConstruct
from copy import deepcopy
import time


class TPreprocess():

    def __init__(self) -> None:
        self.m_attributeType = ATTR_NAME
        self.m_colorSpace = "ref" if ATTR_NAME == 'ref' else "YUV"
        self.m_quanParm = {'qs': 1, 'atq': 1, 'offset': 0}
        self.m_pointCloudIn = TComPointCloud()
        self.m_pointCloudOut = None

    def quantize(self, copy=True):
        if copy:
            self.m_pointCloudOut = deepcopy(self.m_pointCloudIn)
        else:
            self.m_pointCloudOut = self.m_pointCloudIn
        self.m_pointCloudOut.quantization(self.m_quanParm)

    def readOriPointCloud(self, filein):
        self.m_pointCloudIn.readFromFile(filein, self.m_attributeType)
        return self.m_pointCloudIn

    def preprocess(self, copy_ori_pt=False, saveQuanPath=None):
        self.quantize(copy=copy_ori_pt)
        if saveQuanPath:
            self.m_pointCloudOut.saveToFile(saveQuanPath)
        if self.m_colorSpace == "YUV" and self.m_pointCloudOut.m_hasColors:
            self.m_pointCloudOut.convertRGBToYUV()

        return self.m_pointCloudOut

    def getSlice(self, m_pt, max_num=10000000):
        if m_pt.m_numPoint > max_num:
            idx = kdtree_partition(m_pt.m_pos, max_num=max_num)
            m_pointCloudSliceList = []
            for i, id in enumerate(idx):
                m_pointCloudSlice = TComPointCloud()
                m_pointCloudSlice.m_pos = m_pt.m_pos[id].copy()
                m_pointCloudSlice.m_color = m_pt.m_color[id].copy()
                m_pointCloudSlice.m_numPoint = m_pointCloudSlice.m_pos.shape[0]
                m_pointCloudSlice.m_hasColors = m_pt.m_hasColors
                m_pointCloudSlice.m_name = m_pt.m_name.split('.ply')[0] + f'_bk{i:03d}.ply'
                m_pointCloudSliceList.append(m_pointCloudSlice)
            return m_pointCloudSliceList
        return [m_pt]


class TComPointCloud():

    def __init__(self) -> None:
        self.m_pos = np.empty((0, 3), float)
        self.m_color = np.empty((0, 3), int)
        self.m_numPoint = 0
        self.m_attributeType = ''
        self.m_hasColors = False
        self.m_name = ''
        self.m_path = ''

    def sortByIdx(self, idx):
        self.m_pos = self.m_pos[idx]
        self.m_color = self.m_color[idx]

    def getByIdx(self, idx):
        return np.c_[self.m_pos[idx], self.m_color[idx]]

    def readFromFile(self, path, atriType):
        assert os.path.exists(path), 'File not exist: ' + path
        pt_data = ptIO.pcread(path, True)
        self.m_pos = pt_data[0]
        self.m_numPoint = self.m_pos.shape[0]
        assert self.m_pos.ndim == 2 and self.m_pos.shape[1] == 3 and self.m_numPoint > 0, 'Error read ' + path
        if atriType == 'rgb' or atriType == 'ref':
            self.m_attributeType = atriType
            self.m_color = pt_data[1]
            assert self.m_color.shape[0] == self.m_numPoint
            self.m_hasColors = True
        if atriType == "ref":
            self.m_color = np.tile(self.m_color[:, :1], (1, 3))  # to make sure attribute shape is (N,3)
        self.m_name = os.path.basename(path)
        self.m_path = path

    def saveToFile(self, path, asAscii=False):
        if self.m_hasColors:
            assert self.m_attributeType in ['rgb', 'ref'], f'Error m_attributeType: {self.m_attributeType}'
            if self.m_attributeType == 'rgb':
                ptIO.pcwrite(path, self.m_pos, self.m_color, asAscii=asAscii)
            if self.m_attributeType == 'ref':
                ptIO.pcwrite(path, self.m_pos, self.m_color[:, :1], asAscii=asAscii)
        else:
            ptIO.pcwrite(path, self.m_pos)
        print('save file ', path)

    def convertRGBToYUV(self):
        self.m_color = RGB2YCoCg(self.m_color[:, :3])

    def convertYUVToRGB(self):
        self.m_color = YCoCg2RGB(self.m_color[:, :3])

    def clearColors(self):
        self.m_color = np.zeros((self.m_numPoint, 3), int)
        self.m_hasColors = True

    def quantization(self, quan_parm):
        refPt = self.m_pos[:, 0:3]
        if 'offset' in quan_parm:
            offset = quan_parm['offset']
            if quan_parm['offset'] == "min":
                offset = refPt.min(0, keepdims=True)
            if quan_parm['offset'] == "mean":
                offset = refPt.mean(0, keepdims=True)
        else:
            offset = 0
        points = refPt - offset
        if 'qlevel' in quan_parm and quan_parm['qlevel'] is not None:
            quan_parm['qs'] = (points.max() - points.min()) / (2**quan_parm['qlevel'] - 1)
        qpt = np.round(points / quan_parm['qs']).astype(int)
        quanpt, idx = np.unique(qpt, axis=0, return_index=True)
        self.m_pos = quanpt
        if self.m_hasColors:
            self.m_color = np.round(self.m_color * quan_parm['atq'])[idx].astype(int)
        quan_parm['offset'] = offset
        self.m_numPoint = quanpt.shape[0]

    def deQuantization(self, dequan_parm):
        self.m_pos = self.m_pos * dequan_parm['qs'] + dequan_parm['offset']
        self.m_color = np.round(self.m_color / dequan_parm['atq']).astype(int)

    def addPoints(self, pos, color=None):
        self.m_pos = np.vstack((self.m_pos, pos))
        if color is not None:
            self.m_color = np.vstack((self.m_color, color))
            self.m_hasColors = True
        self.m_numPoint += pos.shape[0]


# _________________________________________________________________________________________________
class TContext():

    def __init__(self, pointCloudOrg: TComPointCloud) -> None:
        self.m_pointCloudOrg = pointCloudOrg
        # data
        self.p_m = None  # p_m: xyz,yuv,ptid, sliceID, ispadding
        self.pred_nnid = None
        self.pred_dis = None

        self.gt_target = None
        self.target = None
        self.pred_yuv = None

        self.base_layer = None
        self.infer_layer = None
        self.predictors = None
        self.r_idx = None

    def Base_LOD(self):
        indexes, r_idx, self.predictors = levelOfDetailLayeringStructure(self.m_pointCloudOrg.m_pos)
        # sort the org pointcloud by lod order
        self.m_pointCloudOrg.sortByIdx(indexes)
        self.base_layer = self.m_pointCloudOrg.getByIdx(r_idx > INTRA_LOD_START)
        self.infer_layer = self.m_pointCloudOrg.getByIdx(r_idx <= INTRA_LOD_START)
        self.r_idx = r_idx
        return self.base_layer.astype(int)

    def Infer_LOD(self, bptt=1024):
        self.p_m, self.pred_nnid, self.pred_dis = InferLodConstruct(self.base_layer, self.infer_layer, self.predictors, self.r_idx, self.m_pointCloudOrg.getByIdx(slice(0, None)), K=KNN, bptt=bptt)

    def Update_baselayer(self, base_layer):
        self.base_layer = base_layer
        self.m_pointCloudOrg.m_color[:len(base_layer)] = base_layer[:, 3:6]

    def Update_pred(self, idx=slice(0, None)):
        inv_dis = ((MAX_DIS / (self.pred_dis[idx, 0:KNN]))).astype(int)
        pred_nn = self.p_m[self.pred_nnid[idx, 0:KNN]]
        pred_yuv = pred_from_lod(None, pred_nn, inv_dis)
        if self.pred_yuv is not None:
            self.pred_yuv[idx] = pred_yuv
        else:
            self.pred_yuv = pred_yuv

    def Predict(self):
        self.Update_pred()
        self.gt_target = self.p_m[:, 3:6] - self.pred_yuv
        self.target = np.clip(self.gt_target, -ATTIBUTE_HALF_RANGE, ATTIBUTE_HALF_RANGE)
        self.out_target = self.gt_target - self.target
        self.target += ATTIBUTE_HALF_RANGE

    def construct_context(self, slice_id, MAX_BATCH=48):
        idxs = np.where((self.p_m[:, 7] == -slice_id))[0]
        if self.m_pointCloudOrg.m_attributeType == 'ref':
            kd_idx = kdtree_partition(np.c_[self.p_m[idxs, 0:3], idxs], max_num=1024 * 4, return_all=True)
            idxs = np.concatenate(kd_idx, 0)[:, -2]
            idxs = np.r_[idxs, np.zeros(int(np.ceil(len(idxs) / 1024) * 1024) - len(idxs), )].astype(int)
        total_Batch = len(idxs) // 1024
        idxs = idxs.reshape(total_Batch, 1024).tolist()
        dict_datas = []
        for idx in [idxs[x:MAX_BATCH + x] for x in range(0, len(idxs), MAX_BATCH)]:
            nn_idx = self.pred_nnid[idx, 0:KNN]
            if self.m_pointCloudOrg.m_attributeType == 'ref':
                target = self.target[idx, :1]
                padding = -(np.array(idx) != 0).astype(int)
            else:
                target = self.target[idx, :]
                padding = self.p_m[idx, 8]
            dict_datas.append({
                'geo': self.p_m[idx, :6],
                'target': target,
                'resnn': self.p_m[:, 3:6][nn_idx, :] - self.pred_yuv[nn_idx, :],
                'predint': self.pred_yuv[idx, :],
                'nn': self.p_m[nn_idx, :6],
                'padding': padding,
                'idx': self.p_m[idx, 6]
            })
        return dict_datas

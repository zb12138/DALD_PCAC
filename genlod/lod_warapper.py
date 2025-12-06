'''
Author: chunyangf@qq.com
'''
from ctypes import *
import numpy as np
import os
import sys
import time


def dec2binAry(x, bits=8):
    # array([[128,  64,  32,  16,   8,   4,   2,   1]])
    mask = np.expand_dims(2**np.arange(bits-1, -1, -1), 1).T
    return (np.bitwise_and(np.expand_dims(x, 1), mask) != 0).astype(int)


def bin2decAry(x):
    if (x.ndim == 1):
        x = np.expand_dims(x, 0)
    bits = x.shape[1]
    mask = np.expand_dims(2**np.arange(bits-1, -1, -1), 1)
    return x.dot(mask).astype(int)


def Morton(A):
    A = A.astype(np.int32)
    n = np.ceil(np.log2(np.max(A)+1)).astype(np.int32)  # 
    x = dec2binAry(A[:, 0], n)  # shape: [点数,n] (int)
    y = dec2binAry(A[:, 1], n)
    z = dec2binAry(A[:, 2], n)
    m = np.stack((x, y, z), 2)  # shape: [点数,n,3]
    m = np.transpose(m, (0, 2, 1))  # shape: [点数,3,n]
    mcode = np.reshape(m, (A.shape[0], 3*n), order='F')  # shape: [点数,3*n]
    return mcode


sys.path.append(os.path.dirname(os.path.dirname(__file__)))


# int32_t* inputpoints, int pointNum, int32_t* pointcloud_lod,uint32_t* indexes, uint32_t* numPointsInLod ,  int32_t* predictors
lib = cdll.LoadLibrary(os.path.dirname(os.path.abspath(
    __file__))+'/lod.so')  # class level loading lib
lib.genLod.restype = c_void_p
lib.genLod.argtypes = [POINTER(c_int32), c_uint32, POINTER(
    c_int32), POINTER(c_uint32), POINTER(c_uint32), POINTER(c_int32)]


def genLod(ori_data, return_lod_order=True):
    '''

    '''
    assert ori_data.shape[1] == 6
    data = ori_data.copy()

    # shift and sorted by Morton
    offset = data[:, 0:3].min(0)
    data[:, 0:3] -= offset

    indexes_morton = np.argsort(bin2decAry(Morton(data[:, 0:3])), axis=0)[:, 0]
    data = data[indexes_morton]

    pointNum = data.shape[0]
    data = np.reshape(data, data.shape, order='C')
    data = np.ascontiguousarray(data).astype(np.int32)
    data_p = data.ctypes.data_as(POINTER(c_int32))
    pre_pointcloud_lod = (c_int32*(pointNum*6))()
    indexes = (c_uint32*pointNum)()
    numPointsInLod = (c_uint32*pointNum)()
    predictors = (c_int32*(pointNum*8))()
    lib.genLod(data_p, pointNum, pre_pointcloud_lod,
               indexes, numPointsInLod, predictors)

    pre_pointcloud_lod = np.array(pre_pointcloud_lod).reshape(-1, 6)
    # neighborCount,predMode,predictorIndex,weight
    predictors = np.array(predictors).reshape(-1, 8)
    numPointsInLod = np.array(numPointsInLod)
    assert pointNum in numPointsInLod
    numPointsInLod = numPointsInLod[:int(
        np.where(numPointsInLod == pointNum)[0])]
    indexes = np.array(indexes)
    # //in lod order, xth point's nn is [predictors[x]](lod order)
    sorted_pointcloud = data

    r_id = np.zeros((pointNum,))
    numPointsInLod_temp = [0]+numPointsInLod.tolist()+[pointNum]
    for i in range(1, len(numPointsInLod_temp)):
        r_id[numPointsInLod_temp[i-1]:numPointsInLod_temp[i]] = i

    # all inter lod predition make sure:
    # [r_idx[predictors[r_idx==lod_idx,2:5]].max()<lod_idx for lod_idx in range(2,int(r_idx.max()))]
    if return_lod_order:
        pred_pt = pre_pointcloud_lod[indexes]  # not used
        gt_lodoerder_pt = sorted_pointcloud[indexes]
        pointNN = gt_lodoerder_pt[predictors[:, 2:5]]
        weight = np.transpose(
            (np.expand_dims(predictors, 1)[:, :, 5:8]), (0, 2, 1))
        # not use pre_pointcloud_lod from gpcc
        pred_pt = np.round(((pointNN*weight).sum(1)/weight.sum(1)))
    else:
        rev_indexes = np.zeros_like(indexes)
        rev_indexes[indexes] = np.array(range(pointNum))
        gt_lodoerder_pt = sorted_pointcloud
        predictors[:, 2:5] = indexes[predictors[:, 2:5]]
        predictors = predictors[rev_indexes]
        pointNN = gt_lodoerder_pt[predictors[:, 2:5]]
        weight = np.transpose(
            (np.expand_dims(predictors, 1)[:, :, 5:8]), (0, 2, 1))
        # not use pre_pointcloud_lod from gpcc
        pred_pt = np.round(((pointNN*weight).sum(1)/weight.sum(1)))
        r_id = r_id[rev_indexes]

    # NB:
    # if  return_lod_order==True, return in lod_order and p[indexes]==gt_lodoerder_pt
    # else will return gt_lodoerder_pt in Morton order
    if return_lod_order == False:
        indexes = indexes_morton
    else:
        indexes = indexes_morton[indexes]
    return pred_pt, indexes, r_id, predictors, gt_lodoerder_pt  # return in lod order


if __name__ == "__main__":
    import testTool.pt as pointcloud
    data = pointcloud.pcread('Data/MPEG8i/loot_vox10_1200.ply')
    data[1] = pointcloud.RGB2YCoCg(data[1])
    pointcloud_lod, indexes, numPointsInLod, predictors = genLod(
        np.concatenate(data, -1))
    print(predictors)

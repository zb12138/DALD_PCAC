'''
Author: chunyangf@qq.com
LastEditors: chunyang fu
Description: LOD generation and inference
Date: 2025-12-06 16:20:33
'''
from genlod.lod_warapper import genLod
from networkTool import INTRA_LOD_START, MAX_SLICE_NUM, KNN, MAX_DIS, IS_LIDAR
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import KDTree
import testTool.ptIO as ptIO
import testTool.ptfun as ptfun
import threading


def convertRow2ColIdx(shape, padid):
    row = padid // shape[1]
    col = padid - row * shape[1]
    s1 = col * shape[0] + row
    return s1.astype(int)


def gen_block_by_kmeans(base_layer, infer_layer, block_num, geo_weight=255, bptt=1024, SLICE=MAX_SLICE_NUM):
    tree_base = KDTree(base_layer[:, :3], compact_nodes=False)
    min_dis, min_nnIDx = tree_base.query(base_layer[:, :3], k=50)
    base_layer_ave = base_layer[min_nnIDx, 3:6].mean(1)
    base_ord = base_layer[:, :3].copy().astype(float)
    base_ord = base_ord - base_ord.min()
    base_ord /= base_ord.max()
    kmeans = KMeans(n_clusters=block_num, random_state=0, n_init=10, max_iter=10).fit(np.c_[base_ord[:, :3] * geo_weight, base_layer_ave])
    base_label = kmeans.labels_
    min_dis, min_nnIDx = tree_base.query(infer_layer[:, :3], k=1)
    infer_label = base_label[min_nnIDx]
    label = np.ones((len(infer_layer), 3)).astype(int) * (-1)  # pointid,bkid,ispadding
    label[:, 0] = np.arange(len(infer_layer))
    label[:, 1] = infer_label
    _, sortBks = ptfun.sortByMorton(np.concatenate([base_layer[base_label == x, 0:3].mean(0, keepdims=True) for x in range(block_num)], 0).astype(int), return_idx=True)
    sortBks = np.argsort([base_layer[base_label == x, 0:3].mean() for x in range(block_num)])
    data = []
    remaingdatas = []
    for i in sortBks:
        idx = label[label[:, 1] == i, 0]
        t = len(idx) // (SLICE * bptt) * (SLICE * bptt)
        data.append(label[idx[:t], 0].reshape(-1, SLICE))
        remaingdatas.append(label[idx[t:], 0])
    remaingdata = np.concatenate(remaingdatas, 0)
    remainN = remaingdata.shape[0]
    lackN = int(np.ceil(remainN / (MAX_SLICE_NUM * bptt)) * (MAX_SLICE_NUM * bptt)) - remainN
    IDremaing = np.concatenate((remaingdata, np.ones((lackN, ), int) * -1)).reshape(-1, SLICE)
    IDbyBK = np.concatenate(data, 0)

    # gen infer points with padding
    BLS = np.r_[IDbyBK, IDremaing]  # BLS order: SLICE_PTNUM * SLICE
    slice_ptnum = BLS.shape[0]
    padid = np.where(BLS.reshape(-1) == -1)[0][0] + np.arange(0, lackN)  # locate the first padding data
    s1 = convertRow2ColIdx((slice_ptnum, MAX_SLICE_NUM), padid)  # the padding data in new order (BLS)
    s2 = convertRow2ColIdx((slice_ptnum, MAX_SLICE_NUM), padid - np.ceil(lackN / MAX_SLICE_NUM) * MAX_SLICE_NUM)  # the refer data (remaingdata[-lackN:]) of padding data in new order (BLS)
    BLS = BLS.T.reshape(-1)
    BLS[s1] = BLS[s2]
    infer_padding = infer_layer[BLS]
    all_layer = np.r_[base_layer, infer_padding]
    all_layer[:, 8] = -1
    all_layer[s1 + len(base_layer), 8] = s2 + len(base_layer)
    # infer_padding.reshape(-1,16); each clos is a slice; continued rows are bks
    # a = all_layer.astype(int)
    # assert (a[a[a[:,8]>-1,8]]==a[a[:,8]>-1])[:,:8].all()
    # xyz,yuv,ptid(no use), sliceID (no use), ispadding (>-1 for paading,-1 for not padding)
    return all_layer, slice_ptnum


def levelOfDetailLayeringStructure(pos):
    ptNum = pos.shape[0]
    p = np.c_[pos, np.zeros((ptNum, 3))]
    pred_pt, indexes, r_idx, predictors, gt_lodoerder_pt = genLod(p, return_lod_order=True)
    r_idx = r_idx.max() - r_idx + 1
    return indexes, r_idx, predictors


def pred_from_lod(p_m, pred_nn, pred_dis_inv):
    nn_color = pred_nn[:, :, 3:6]
    dis = pred_dis_inv
    pred_color = (dis[..., None].transpose(0, 2, 1) @ nn_color).squeeze(1)
    pred_color_norm = np.round(pred_color / dis.sum(1, keepdims=True)).astype(int)
    return pred_color_norm


def InferLodConstruct(base_layer, infer_layer, predictors, r_idx, gt_lodoerder_pt, K=KNN, bptt=1024):
    if IS_LIDAR:
        return InferLodConstructLiDAR(base_layer, infer_layer, predictors, r_idx, gt_lodoerder_pt, K=KNN, bptt=bptt)
    else:
        return InferLodConstructObj(base_layer, infer_layer, predictors, K=KNN, bptt=bptt)


def InferLodConstructLiDAR(base_layer, infer_layer, predictors, r_idx, gt_lodoerder_pt, K=KNN, bptt=1024):
    base_layer = np.c_[base_layer, np.zeros((base_layer.shape[0], 3))]
    infer_layer = np.c_[infer_layer, np.zeros((infer_layer.shape[0], 3))]
    ptNum = gt_lodoerder_pt.shape[0]
    p_m = np.c_[gt_lodoerder_pt, np.arange(ptNum).reshape(-1, 1), r_idx].astype(int)

    is_last_lod = p_m[:, 7] <= INTRA_LOD_START
    is_first_lod = p_m[:, 7] > INTRA_LOD_START

    pred_nnid = np.zeros((ptNum, K), int)
    pred_dis = np.ones((ptNum, K)) * MAX_DIS

    dis = ((p_m[predictors[is_first_lod, 2:5], 0:3] - np.expand_dims(p_m[is_first_lod, 0:3], 1))**2).sum(2)
    pred_dis[is_first_lod, 0:3] = dis
    pred_nnid[:, :3] = predictors[:, 2:5]

    # gen slice from the last_lod in morton
    last_lod = p_m[is_last_lod]
    p_m = np.r_[p_m[is_first_lod], last_lod]
    p_m[:, 6] = np.arange(ptNum)
    last_lod = p_m[p_m[:, 7] <= INTRA_LOD_START]
    # update and find the nn_idx for the last lod
    new_idx = p_m[is_first_lod, 6]

    for i in range(MAX_SLICE_NUM):
        temp_fine = last_lod[i::MAX_SLICE_NUM]
        new_idx = np.r_[new_idx, temp_fine[:, 6]]
        gloabl = p_m[(p_m[:, 7] > INTRA_LOD_START) + (p_m[:, 7] <= 0)]
        tree_gloabl = KDTree(gloabl[:, :3], compact_nodes=False)
        min_dis, min_nnIDx = tree_gloabl.query(temp_fine[:, :3], K, workers=-1)
        pred_nnid[temp_fine[:, 6]] = gloabl[min_nnIDx, 6]
        pred_dis[temp_fine[:, 6]] = min_dis**2
        p_m[temp_fine[:, 6], 7] = -i

    rev_indexes = np.zeros_like(new_idx)
    rev_indexes[new_idx] = np.array(range(new_idx.shape[0]))
    pred_nnid = rev_indexes[pred_nnid[new_idx]]
    p_m = p_m[new_idx]
    p_m[:, 6] = np.arange(ptNum)
    pred_dis = pred_dis[new_idx]

    pred_dis[0, :] = 1

    return p_m.astype(int), pred_nnid, pred_dis.astype(int)


def InferLodConstructObj(base_layer, infer_layer, predictors, K=KNN, bptt=1024):
    base_layer = np.c_[base_layer, np.zeros((base_layer.shape[0], 3))]
    infer_layer = np.c_[infer_layer, np.zeros((infer_layer.shape[0], 3))]
    infer_padding, slice_ptnum = gen_block_by_kmeans(base_layer, infer_layer, max(1, len(infer_layer) // (MAX_SLICE_NUM * 4096)), SLICE=MAX_SLICE_NUM, bptt=bptt)
    p_m = infer_padding
    base_ptnum = len(base_layer)
    ptNum = p_m.shape[0]
    p_m[:, 6] = np.arange(ptNum)
    gloablpt = p_m[:len(base_layer), :]
    pred_nnid = np.zeros((ptNum, K), int)
    pred_dis = np.ones((ptNum, K)) * MAX_DIS * 2
    pred_nnid[:base_ptnum, :3] = predictors[:base_ptnum, 2:5]
    dis = ((p_m[predictors[:base_ptnum, 2:5], 0:3] - np.expand_dims(p_m[:base_ptnum, 0:3], 1))**2).sum(2)
    pred_dis[:base_ptnum, 0:3] = dis

    # search nearest neighbors in multi-thread
    args = []
    threads = []
    results = []

    def process_slice(temp_fine, gloabl, idx_in_slice):
        tree_gloabl = KDTree(gloabl[:, :3], compact_nodes=False)
        min_dis, min_nnIDx = tree_gloabl.query(temp_fine[:, :3], K, workers=-1)
        return gloabl[min_nnIDx, 6], min_dis, idx_in_slice

    for i in range(MAX_SLICE_NUM):
        idx_in_slice = slice(base_ptnum + i * slice_ptnum, base_ptnum + i * slice_ptnum + slice_ptnum)
        temp_fine = p_m[idx_in_slice]
        p_m[idx_in_slice, 7] = -i
        args.append((temp_fine, gloablpt, idx_in_slice))
        gloablpt = np.r_[gloablpt, temp_fine]
    for i in range(MAX_SLICE_NUM):
        t = threading.Thread(target=lambda: results.append(process_slice(*args[i])))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()

    for pred_nnid_r, min_dis, idx_in_slice in results:
        pred_nnid[idx_in_slice] = pred_nnid_r
        pred_dis[idx_in_slice] = min_dis**2

    p_m[:base_layer.shape[0], 7] = 1
    pred_dis = np.clip(pred_dis, 1, MAX_DIS)
    return p_m.astype(int), pred_nnid, pred_dis.astype(int)


# enable this function to debug
# def InferLodConstruct(base_layer, infer_layer, predictors, K=KNN, bptt=1024):
#     base_layer = np.c_[base_layer, np.zeros((base_layer.shape[0], 3))]
#     infer_layer = np.c_[infer_layer, np.zeros((infer_layer.shape[0], 3))]

#     infer_padding, slice_ptnum = gen_block_by_kmeans(base_layer,
#                                                      infer_layer,
#                                                      max(1,len(infer_layer) //
#                                                      (MAX_SLICE_NUM * 4096)),
#                                                      SLICE=MAX_SLICE_NUM, bptt=bptt)
#     p_m = infer_padding
#     base_ptnum = len(base_layer)
#     ptNum = p_m.shape[0]
#     p_m[:, 6] = np.arange(ptNum)
#     gloabl = p_m[:len(base_layer), :]
#     pred_nnid = np.zeros((ptNum, K), int)
#     pred_dis = np.ones((ptNum, K)) * MAX_DIS*2
#     pred_nnid[:base_ptnum, :3] = predictors[:base_ptnum, 2:5]
#     dis = ((p_m[predictors[:base_ptnum, 2:5], 0:3] -
#             np.expand_dims(p_m[:base_ptnum, 0:3], 1))**2).sum(2)
#     pred_dis[:base_ptnum, 0:3] = dis

#     for i in range(MAX_SLICE_NUM):
#         idx_in_slice = slice(base_ptnum + i * slice_ptnum,
#                              base_ptnum + i * slice_ptnum + slice_ptnum)
#         temp_fine = p_m[idx_in_slice]
#         tree_gloabl = KDTree(gloabl[:, :3], compact_nodes=False)
#         min_dis, min_nnIDx = tree_gloabl.query(temp_fine[:, :3], K, workers=-1)
#         pred_nnid[idx_in_slice] = gloabl[min_nnIDx, 6]
#         p_m[idx_in_slice, 7] = -i
#         pred_dis[idx_in_slice] = min_dis**2
#         gloabl = np.r_[gloabl, temp_fine]
#     p_m[:base_layer.shape[0], 7] = 1
#     pred_dis = np.clip(pred_dis,1,2*MAX_DIS)
#     return p_m.astype(int), pred_nnid, pred_dis.astype(int)

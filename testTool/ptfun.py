from subprocess import PIPE
from plyfile import PlyData, PlyElement
import subprocess
import os
import h5py
import numpy as np
import open3d as o3d
import re

PCERRORPATH = "testTool/pc_error"


class CPrintl():

    def __init__(self, logName) -> None:
        self.log_file = logName
        if os.path.dirname(logName) != '':
            os.makedirs(os.path.dirname(logName), exist_ok=True)

    def __call__(self, *args, sep=' '):
        print(*args, sep=sep)
        print(*args, file=open(self.log_file, 'a'), sep=sep)


def RGB2YCoCg(rgb, offset=np.array([0, 256, 256])):
    rgb = rgb.astype(np.int32)
    R, G, B = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    Co = R - B
    t = B + (Co >> 1)
    Cg = G - t
    Y = t + (Cg >> 1)
    return np.hstack((Y, Co, Cg)) + offset.T


def YCoCg2RGB(YCoCg, offset=np.array([0, 256, 256])):
    YCoCg = YCoCg - offset.T
    Y, Co, Cg = YCoCg[:, 0:1], YCoCg[:, 1:2], YCoCg[:, 2:3]
    t = Y - (Cg >> 1)
    G = Cg + t
    B = t - (Co >> 1)
    R = B + Co
    return np.hstack((R, G, B)).astype(np.uint8)


def pcerror(pcRefer, pc, pcReferNorm, pcerror_cfg_params, pcerror_result=PIPE, pcerror_path=PCERRORPATH):
    '''
    Options: 
            --help=0            This help text
      -a,   --fileA=""          Input file 1, original version
      -b,   --fileB=""          Input file 2, processed version
      -n,   --inputNorm=""      File name to import the normals of original point
                                cloud, if different from original file 1n
      -s,   --singlePass=0      Force running a single pass, where the loop is
                                over the original point cloud
      -d,   --hausdorff=0       Send the Haursdorff metric as well
      -c,   --color=0           Check color distortion as well
      -l,   --lidar=0           Check lidar reflectance as well
      -r,   --resolution=0      Specify the intrinsic resolution
            --dropdups=2        0(detect), 1(drop), 2(average) subsequent points
                                with same coordinates
            --neighborsProc=1   0(undefined), 1(average), 2(weighted average),
                                3(min), 4(max) neighbors with same geometric
                                distance
            --averageNormals=1  0(undefined), 1(average normal based on neighbors
                                with same geometric distance)
            --mseSpace=1        Colour space used for PSNR calculation
                                0: none (identity) 1: ITU-R BT.709 8: YCgCo-R
            --nbThreads=1       Number of threads used for parallel processing
    '''
    TEMPPATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "pc_error/temp")
    if not os.path.exists(TEMPPATH) and (type(pc) is not str or type(pcRefer) is not str):
        os.makedirs(TEMPPATH)

    if type(pcerror_result) is str:
        pcLabel = os.path.basename(pcerror_result).split(".")[0]
    else:
        pcLabel = "pt0"
    if type(pc) is not str:
        pcwrite(TEMPPATH + pcLabel + "pc.ply", pc)
        pc = TEMPPATH + pcLabel + "pc.ply"
    if type(pcRefer) is not str:
        pcwrite(TEMPPATH + pcLabel + "pcRefer.ply", pcRefer)
        pcRefer = TEMPPATH + pcLabel + "pcRefer.ply"

    if pcerror_result is not None:
        if type(pcerror_result) is int:
            f = pcerror_result
        else:
            f = open(pcerror_result, 'a+')

    else:
        import sys
        f = sys.stdout
    if type(pcerror_cfg_params) is str:
        pcerror_cfg_params = pcerror_cfg_params.split(' ')
    if pcReferNorm == None:
        pro = subprocess.Popen([pcerror_path, '-a', pcRefer, '-b', pc] + pcerror_cfg_params, stdout=f, stderr=f)
    else:
        pro = subprocess.Popen([pcerror_path, '-a', pcRefer, '-b', pc, '-n', pcReferNorm] + pcerror_cfg_params, stdout=f, stderr=f)
    if pcerror_result == PIPE:
        pro.wait()
        strs = pro.stdout.read().decode('gbk')
        return strs
    else:
        return pro


def voxels2pt(vols, thres=1, scale=1):
    vols = np.squeeze(vols)
    assert vols.ndim == 3
    mask = np.vstack(np.where(vols >= thres)).T
    return mask.astype("float") * scale


def pt2voxels(set_points, cube_size):
    """Transform points to voxels (binary occupancy map).
    Args: points list; cube size;

    Return: A tensor with shape [batch_size, cube_size, cube_size, cube_size, 1]
    """
    if isinstance(set_points, list):
        voxels = []
        for _, points in enumerate(set_points):
            points = points.astype("int")
            vol = np.zeros((cube_size, cube_size, cube_size))
            vol[points[:, 0], points[:, 1], points[:, 2]] = 1.0
            vol = np.expand_dims(vol, 0)
            voxels.append(vol)
        voxels = np.array(voxels)
        return voxels
    else:
        points = set_points.astype("int")
        vol = np.zeros((cube_size, cube_size, cube_size))
        vol[points[:, 0], points[:, 1], points[:, 2]] = 1.0
        vol = np.expand_dims(vol, 0)
        voxels = np.array(vol)
        return voxels


def downsample(pt, sample=1.0):
    if sample < 1:
        xyz = pt
        idx = np.random.choice(xyz.shape[0], int(xyz.shape[0] * sample))
        pt = xyz[idx, :]
    return pt


def readFord(file_path):
    # bug for PlyData read ford because unexpeceted ' ' in 'end_header'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    content = content.replace('end_header ', 'end_header')
    output_file = '/tmp/Ford.ply'
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(content)
    return output_file


def pcshow(points, colors=None, normals=None):
    print('press H for more help.')
    print('press N for Turn on/off point cloud normal rendering.')

    point_cloud = o3dPointCloud(points, colors=colors, normals=normals)
    o3d.visualization.draw([point_cloud], non_blocking_and_return_uid=True)


def o3dPointCloud(points, colors=None, normals=None):
    if points.shape[1] == 6:
        colors = points[:, 3:6]
        points = points[:, 0:3]
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        point_cloud.colors = o3d.utility.Vector3dVector(colors / 255)
    if normals is not None:
        point_cloud.normals = o3d.utility.Vector3dVector(normals)
    return point_cloud


def bin2decAry_numpy(x):
    x = np.asarray(x, dtype=np.float64)
    bits = x.shape[1]
    mask = 2**np.arange(bits - 1, -1, -1).reshape(1, -1)
    return (x * mask).sum(axis=1)


def compute_morton_code_numpy(points):
    if len(points) == 0:
        return np.array([])
    points = points.astype(np.int64)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    ptnum = points.shape[0]
    n = int(np.ceil(np.log2(np.max(points) + 1)))
    morton_code = np.zeros((ptnum, 3 * n), dtype=np.int32)
    for i in range(n):
        morton_code[:, 3 * i] = (z >> i) & 1
        morton_code[:, (3 * i + 1)] = (y >> i) & 1
        morton_code[:, (3 * i + 2)] = (x >> i) & 1
    morton_code = np.flip(morton_code, axis=1)
    return bin2decAry_numpy(morton_code)


def sortByMorton(points, return_idx=False):
    data = points.copy()
    offset = data[:, 0:3].min(0)
    data[:, 0:3] -= offset
    morton_codes = compute_morton_code_numpy(data.astype(np.int64))
    idx = np.argsort(morton_codes)
    if return_idx:
        return points[idx], idx
    else:
        return points[idx]


def write_ply_data(filename, points, attributeName=[], attriType=[]):
    '''
    write data to ply file.
    e.g pt.write_ply_data('ScanNet_{:5d}.ply'.format(idx), np.hstack((point,np.expand_dims(label,1) )) , attributeName=['intensity'], attriType =['uint16'])
    '''
    if type(points) is list:
        points = np.array(points)

    attrNum = len(attributeName)
    assert points.shape[1] >= (attrNum + 3)
    if os.path.dirname(filename) != '' and not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    plyheader = ['ply\n', 'format ascii 1.0\n'] + ['element vertex ' + str(points.shape[0]) + '\n'] + ['property float x\n', 'property float y\n', 'property float z\n']
    for aN, attrName in enumerate(attributeName):
        plyheader.extend(['property ' + attriType[aN] + ' ' + attrName + '\n'])
    plyheader.append('end_header')
    typeList = {'uint16': "%d", 'float': "float", 'uchar': "%d"}
    np.savetxt(filename, points, newline="\n", fmt=["%f", "%f", "%f"] + [typeList[t] for t in attriType], header=''.join(plyheader), comments='')
    return


def kdtree_partition(pc, max_num=100000, return_all=False):
    indx = []
    pc = np.c_[pc, np.arange(pc.shape[0])]

    class KD_node:

        def __init__(self, point=None, LL=None, RR=None):
            self.point = point
            self.left = LL
            self.right = RR

    def createKDTree(root, data):
        if len(data) <= max_num or ((len(data) % 2 == 1) and len(data) <= max_num + 1) or ((len(data) % 2 == 0) and len(data) <= max_num + 2):
            if return_all:
                indx.append(data)
            else:
                indx.append(data[:, -1].astype(int))
            return
        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]
        # data2_sorted = data2[np.lexsort(data2.T[dim_index, None])]
        point = data_sorted[int(len(data) / 2)]
        root = KD_node(point)
        root.left = createKDTree(root.left, data_sorted[:int((len(data) / 2))])
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):])
        return root

    init_root = KD_node(None)
    _ = createKDTree(init_root, pc)
    return indx

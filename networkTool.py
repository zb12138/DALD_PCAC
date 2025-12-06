'''
Author: chunyangf@qq.com
LastEditors: fcy
Description: Network parameters and helper functions
FilePath: /compression/networkTool.py
'''
from glob import glob
import torch
import os
import random
import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print('use', device)

# Network parameters
IS_LIDAR = True
ATTR_NAME = 'ref' if IS_LIDAR else 'rgb'
MAX_NUM_POINTS = 8192 * 4
INTRA_LOD_START = 2
ATTIBUTE_RES_RANGE = 511
ATTIBUTE_HALF_RANGE = 255
MAX_DIS = 2**(40)
BPTT = 1024
KNN = 9

if IS_LIDAR:
    EXPNAME = 'Exp/fordq1mm'
    expComment = 'lidar ford point cloud compression'
    CKPT_PATH = 'modelsave/fordq1mmK9_encoder_epoch_70003520.pth'
    MAX_SLICE_NUM = 4
else:
    EXPNAME = 'Exp/obj'
    expComment = 'object point cloud compression'
    CKPT_PATH = 'modelsave/RWTT_encoder_epoch_70035840.pth' 
    # CKPT_PATH = 'modelsave/PCL_PCD_Sketchfab_encoder_epoch_70031360.pth'
    # CKPT_PATH = 'modelsave/CNeT_crossYUV_encoder_epoch_30093760.pth'
    MAX_SLICE_NUM = 16

checkpointPath = EXPNAME + '/checkpoint'
# Random seed
seed = 2
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

os.environ["H5PY_DEFAULT_READONLY"] = "1"
# Tool functions


def save(index, saveDict, modelDir='checkpoint', pthType='epoch', keep_pth_num=10):
    savePath = modelDir + '/encoder_{}_{:08d}.pth'
    if os.path.dirname(savePath) != '' and not os.path.exists(os.path.dirname(savePath)):
        os.makedirs(os.path.dirname(savePath))
    torch.save(saveDict, modelDir + '/encoder_{}_{:08d}.pth'.format(pthType, index))
    dir_list = glob(modelDir + '/*.pth')
    if len(dir_list) > keep_pth_num:
        dir_list = sorted(dir_list, key=lambda x: os.path.getmtime(x))
        for i in dir_list[:-keep_pth_num]:
            os.remove(i)


def reload(checkpoint, modelDir='checkpoint', pthType='epoch', print=print, multiGPU=False):
    try:
        if checkpoint is not None:
            saveDict = torch.load(modelDir + '/encoder_{}_{:08d}.pth'.format(pthType, checkpoint), map_location=device)
            pth = modelDir + '/encoder_{}_{:08d}.pth'.format(pthType, checkpoint)
        if checkpoint is None:
            saveDict = torch.load(modelDir, map_location=device)
            pth = modelDir
        saveDict['path'] = pth
        # print('load: ',pth)
        if multiGPU:
            from collections import OrderedDict
            state_dict = OrderedDict()
            new_state_dict = OrderedDict()
            for k, v in saveDict['encoder'].items():
                name = k[7:]  # remove `module.`
                state_dict[name] = v
            saveDict['encoder'] = state_dict
        # print('load: ',pth)
        return saveDict
    except Exception as e:
        print('**warning**', e, ' start from initial model')
        # saveDict['path'] = e
    return None


class CPrintl():
    def __init__(self, logName) -> None:
        self.log_file = logName
        if os.path.dirname(logName) != '' and not os.path.exists(os.path.dirname(logName)):
            os.makedirs(os.path.dirname(logName))

    def __call__(self, *args, end='\n'):
        print(*args, end=end)
        print(*args, file=open(self.log_file, 'a'), end=end)


def model_structure(model, print=print):
    print('-' * 120)
    print('|' + ' ' * 30 + 'weight name' + ' ' * 31 + '|' + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' + ' ' * 3 + 'number' + ' ' * 3 + '|')
    print('-' * 120)
    num_para = 0
    for _, (key, w_variable) in enumerate(model.named_parameters()):
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        num_para += each_para

        print('| {:70s} | {:30s} | {:10d} |'.format(key, str(w_variable.shape), each_para))
    print('-' * 120)
    print('The total number of parameters: ' + str(num_para))
    print('-' * 120)

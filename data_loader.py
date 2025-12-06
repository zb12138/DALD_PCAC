import os
import sys
import glob
import time
from tqdm import tqdm
import numpy as np
import h5py
import torch
import torch.utils.data
from Common import TContext, TPreprocess
from networkTool import MAX_SLICE_NUM
import open3d as o3d
import numpy as np

SUFFLE_FILE_RANGE = 5
GPU_NUM = 1


class PCDataset(torch.utils.data.Dataset):

    def __init__(self, files, bptt, batch_size, num_workers, shuffle=True):
        self.files = []
        self.dataNames = sorted(files)
        self.fileLen = len(self.dataNames)
        self.startFileIdx = list(range(0, self.fileLen, self.fileLen // (num_workers * GPU_NUM)))[:num_workers * GPU_NUM]  # where slice starts.
        self.endFileIdx = self.startFileIdx[1:] + [self.fileLen]  # where slice ends.
        self.fileIndx = self.startFileIdx.copy()
        self.filedir = None
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.numberWorker = num_workers
        self.bptt = bptt
        self.dataBuffer = []
        self.dataLenPerFile = self.getdataLenPerFile()  # skfab dataset 13637711
        self.index = 0
        self.datalen = 0

        if (self.shuffle):
            np.random.shuffle(self.dataNames)

    def getdataLenPerFile(self):
        # return 1327961.7 # return self.dataLenPerFile directly here after you got this value
        self.dataLenPerFile = 0
        fileID = np.random.randint(0, self.fileLen, 5)
        for i in fileID:
            pcd = o3d.io.read_point_cloud(self.dataNames[i])
            self.dataLenPerFile += len(pcd.points)
        self.dataLenPerFile /= 5
        print('dataLenPerFile', self.dataLenPerFile, 'you can reutrn it directly after you see this value')
        return self.dataLenPerFile

    def __len__(self):
        return int(self.dataLenPerFile * self.fileLen / self.bptt)

    def __getitem__(self, index):
        workerID = (index // GPU_NUM) // self.batch_size % self.numberWorker + index % GPU_NUM * self.numberWorker
        if (self.index >= self.datalen):
            self.index = 0
            self.dataBuffer = []
            for _ in range(SUFFLE_FILE_RANGE):
                filename = self.dataNames[self.fileIndx[workerID]]
                try:
                    d = read_file(filename, self.bptt)
                except:
                    print('warning********read', filename, 'error!')
                    d = read_file(self.dataNames[1], self.bptt)
                self.dataBuffer.extend(d)
                self.fileIndx[workerID] += 1  # shuffle step = 1, will load continuous mat
                if (self.fileIndx[workerID] >= self.endFileIdx[workerID]):
                    if (self.shuffle):
                        np.random.seed(2)
                        np.random.shuffle(self.dataNames)
                    self.fileIndx[workerID] = self.startFileIdx[workerID]

            if (self.shuffle):
                np.random.shuffle(self.dataBuffer)
            self.datalen = len(self.dataBuffer)
            self.index = 0

        # try read
        pc_data = self.dataBuffer[self.index]  # (bptt, 4, 6)
        self.index += 1
        return pc_data


def read_file(filename, bptt):
    PointCloudReader = TPreprocess()
    # m_pt = TComPointCloud()
    # m_pt.readFromFile(filename, atriType='rgb')
    # m_pt.convertRGBToYUV()
    # m_pt.quantization({'qs': 1, 'atq': 1, 'offset': 'min'})
    PointCloudReader.readOriPointCloud(filename)
    context = TContext(PointCloudReader.preprocess())
    context.Base_LOD()
    context.Infer_LOD(bptt=bptt)
    context.Predict()
    data = []
    for slice_id in range(MAX_SLICE_NUM):
        data.extend(context.construct_context(slice_id, MAX_BATCH=1))
    return data


def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=True, collate_fn=None):
    g = torch.Generator()
    g.manual_seed(0)
    args = {'batch_size': batch_size, 'num_workers': num_workers, 'collate_fn': collate_fn, 'pin_memory': False, 'drop_last': False, 'generator': g, 'shuffle': shuffle}
    loader = torch.utils.data.DataLoader(dataset, **args)
    return loader


if __name__ == "__main__":
    num_workers = 1
    bptt = 1024  # the number of the continuous occupancy code in data, bptt*batch_size divisible by batchSize
    batch_size = 32
    dataset = glob.glob('Data/8i/longdress/Ply/*.ply') + \
        glob.glob('Data/8i/soldier/Ply/*.ply') + \
        glob.glob('Data/train/Owlii/*/*.ply') + \
        glob.glob('Data/train/MVUB/andrew*/ply/*.ply') +\
        glob.glob('Data/train/MVUB/david*/ply/*.ply') +\
        glob.glob('Data/train/MVUB/sarah*/ply/*.ply')
    # glob.glob('Data/MPEGCat1A/Facade*.ply')*100 +\
    # glob.glob('Data/MPEGCat1A/House_without_roof_00057_vox12*.ply')*100

    filedirs = ['Data/train/8iVFBv2/8iVFBv2/soldier/Ply/soldier_vox10_0777.ply']  # sorted(dataset)
    train_set = PCDataset(files=filedirs, bptt=bptt, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    train_loader = make_data_loader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, repeat=False)
    import tqdm
    bar = tqdm.tqdm(total=len(train_loader))
    a = []
    for batch, d in enumerate(train_loader):
        print(d['geo'].shape)
        bar.update(1)
        pass

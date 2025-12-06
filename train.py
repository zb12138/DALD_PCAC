'''
Author: chunyangf@qq.com
LastEditors: chunyang fu
Description: DALD-PCAC trainer file
Date: 2025-12-06 17:59:37
All rights reserved.
'''
import datetime
import math
import os
import time
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from attentionModel import TransformerModule
from networkTool import EXPNAME, ATTIBUTE_RES_RANGE, BPTT, KNN, device, reload, save, CPrintl, checkpointPath, IS_LIDAR
if IS_LIDAR:
    from DALD_PCAC import DALD_LiDAR as PCAC_Model
else:
    from DALD_PCAC import DALD_RGB as PCAC_Model

Model = PCAC_Model().to(device)

writer = SummaryWriter('./log/' + EXPNAME + datetime.datetime.now().strftime('%m%d'))
printl = CPrintl(EXPNAME + '/train.log')


class Train_and_Test():

    def __init__(self, model_path=None, model=Model) -> None:
        self.model = model
        self.print = print
        # if Model is None:
        self.model_path = model_path
        if model_path:
            saveDic = reload(None, model_path, print=self.print)
            assert saveDic is not None
            self.model.load_state_dict(saveDic['encoder'])
            print("retrain from: ", model_path)

        lr = 1e-3  # learning rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, 1.0, gamma=0.95)

        self.criterion = nn.CrossEntropyLoss()
        self.log_interval = 32
        self.iteration = 0
        self.loginter = 0
        self.accbpp = 0
        self.ce_loss = 0
        self.loss = None
        self.epoch = 0
        self.train_loader = None

        # for test
        self.net_acbit = 0
        self.net_celoss = 0
        self.net_ptNum = 0
        self.net_elapsed = 0

    def train_one_epoch(self, epoch):
        self.ce_loss, self.accbpp = 0., 0.
        for Batch, d in enumerate(((self.train_loader))):
            self.optimizer.zero_grad()
            for k in d.keys():
                if k == 'idx':
                    continue
                d[k] = d[k].squeeze(1).to(device)
            self.network(d, test_type='train')

            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.iteration)

            if self.loginter > self.log_interval:
                cur_ce_loss = self.ce_loss / self.loginter
                printl('| epoch {:3d} | {:4d}/{:4d} batches | ms/batch {:5.2f} | iteration {:8d} | CE loss {:5.2f}'.format(epoch, Batch, len(self.train_loader),
                                                                                                                           self.net_elapsed * 1000 / self.loginter, self.iteration, cur_ce_loss))
                writer.add_scalar('ce_loss', cur_ce_loss, self.iteration)  # Cross-entropy loss
                self.ce_loss = 0.
                self.net_elapsed = 0
                self.loginter = 0
            if Batch % (self.log_interval * 10) == 0:
                save(epoch * 10000000 + Batch, saveDict={'epoch': self.epoch, 'iteration': self.iteration, 'encoder': self.model.state_dict()}, modelDir=checkpointPath)

    def train(self, epochs, train_loader):
        self.model.train()  # Turn on the train mode
        self.train_loader = train_loader
        for epoch in range(epochs):
            self.train_one_epoch(epoch)
        printl('-' * 89)
        self.scheduler.step()
        printl('-' * 89)

    def network(self, data, test_type):
        target = data['target']
        assert data['geo'].shape[0] > 0
        assert data['geo'].shape[1] > 10
        start = time.time()
        output = self.model(data, target)
        self.net_elapsed += (time.time() - start)

        if test_type == 'train' or test_type == 'encode':
            loss = self.criterion(output.reshape(-1, ATTIBUTE_RES_RANGE), target.reshape(-1)) / math.log(2)
        if test_type == 'train':
            self.loss = (loss)  #/bpp
            self.ce_loss += self.loss.item()
            self.iteration += 1
            self.loginter += 1
        else:
            if test_type == 'encode':
                pointNum = data['geo'].shape[0] * data['geo'].shape[1]
                self.net_ptNum += pointNum
                self.net_acbit += (loss).item() * pointNum * 3
                output = torch.softmax(output.detach(), dim=-1)
                return output
            if test_type == 'decode':
                pointNum = data['geo'].shape[0] * data['geo'].shape[1]
                self.net_ptNum += pointNum
                output = torch.softmax(output.detach(), dim=-1)
                return output


if __name__ == "__main__":
    import glob
    import os
    import time
    from data_loader import PCDataset, make_data_loader
    num_workers = 4
    bptt = BPTT
    batch_size = 32

    if IS_LIDAR:
        dataset = glob.glob('Data/MPEG/MPEGCat3Frame/Ford_01_q1mm/*.ply')
    else:
        dataset = glob.glob('Data/RWTT/point_cloud/*.ply')

    filedirs = sorted(dataset)
    train_set = PCDataset(files=filedirs, bptt=bptt, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    train_loader = make_data_loader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, repeat=False)
    Ctrain = Train_and_Test(model_path='')
    printl(datetime.datetime.now().strftime('\r\n%Y-%m-%d:%H:%M:%S'))
    printl('Pid: ' + str(os.getpid()))
    Ctrain.train(epochs=8, train_loader=train_loader)

'''
Author: chunyangf@qq.com
LastEditors: chunyang fu
Description: NN metric testing file
Date: 2025-12-06 16:20:33
All rights reserved.
'''
import MinkowskiEngine as ME
import torch
import glob
import os

k_size = 5
channels = 1


def densityTest(xyz):
    geo = ME.utils.batched_coordinates([xyz])
    input = ME.SparseTensor(features=torch.ones((geo.shape[0], 1)), coordinates=geo)
    Conv_fun = ME.MinkowskiConvolution(in_channels=channels, out_channels=3, kernel_size=k_size, dimension=3)
    Conv_fun.kernel.data = torch.ones_like(Conv_fun.kernel.data)
    with torch.no_grad():
        output = Conv_fun(input)
        return output.F.cpu().numpy().mean()


if __name__ == '__main__':
    from pt import ptread
    ave = 0
    cnt = 0
    files = glob.glob('ford_q20mm/*/*.ply')
    # create geo file
    for inputFile in files:
        if inputFile.endswith('n.ply'):
            continue
        try:
            xyz = ptread(inputFile)
            densityNN = densityTest(xyz)
            ave += densityNN
            cnt += 1
            print(os.path.basename(inputFile), ' ', densityNN)
        except:
            print(inputFile, 'error')
            continue
    print(ave / cnt)

import pt, glob, os, sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Common import TComPointCloud

## for Sketchfab
# file = sorted(glob.glob('Data/Sketchfab/0.quantilized/*.ply'))
# for f in file:
#     p, c = pt.pcread(f)
#     indx = pt.kdtree_partition(p, max_num=1000000)
#     print(f)
#     for i, id in enumerate(indx):
#         pt.pcwrite('Data/Sketchfab/partitioned/' + os.path.basename(f)[:-4] + f'bk{i:03d}.ply', p[id], c[id])

## for semanticKITTI
## Note that, calc bpip for this data
vox = 12
seq = '11'
file = sorted(glob.glob(f'Data/semanticKITTI/dataset/sequences/{seq}/velodyne/*.bin'))
for f in file:
    # p,c = pt.pcread(f)
    lidar = TComPointCloud()
    lidar.readFromFile(f, 'ref')
    lidar.quantization({'qs': 400 / (2**vox - 1), 'offset': -200, 'atq': 100})
    lidar.saveToFile(f'semanticKITTI_vox{vox}/{seq}/' + os.path.basename(f)[:-3] + 'ply')

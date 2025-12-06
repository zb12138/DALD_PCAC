# sampling point
from testTool.pt import pcread, pcwrite
from testTool.MinKovDenstiyTest import densityTest
from EncTop import TEncTop, CPrintl, TestFile
import numpy as np

printl = CPrintl('Facade_00064_vox11.log')
p, c = pcread('Data/MPEG/MPEGCat1A/Facade_00064_vox11.ply')
for ratio in [0.01, 0.05, 0.1, 0.2, 0.5, 0.8]:
    ptNum = p.shape[0]
    idx = np.random.permutation(ptNum)[:round(ptNum * ratio)]
    denstiy = densityTest(p[idx, :])
    print(ratio, denstiy)
    f = f'Facade_00064_vox11/sample{ratio}_{denstiy}.ply'
    pcwrite(f, p[idx, :], c[idx, :], saveInfloat=True)

    encoder = TEncTop()
    test = TestFile(path=f)
    result = test.testByFun(encoding_fun=lambda x: encoder.encode(x), print=printl)
    print(result)

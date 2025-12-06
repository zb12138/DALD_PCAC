'''
Author: chunyangf@qq.com
'''
import re
import glob

EXPNAME = 'Exp/GPCC'
from tqdm import tqdm
import pandas as pd
from termcolor import cprint
import subprocess
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import testTool.ptIO as ptIO

TMC_PATH = 'testTool/tmc13v23'
PCERROR_PATH = "testTool/pc_error"


def parse_cmd_output(strs, type='tmc13Encode', attriName="colors"):

    def parse_helper(head, head2="\n", count=0):
        try:
            return np.array(float(strs.split(head)[count + 1].split(head2)[0]))
        except:
            return np.array(np.nan)

    if type in ['tmc13Encode']:
        SliceNum = int(parse_helper('Slice number: '))
        positions_byte, positions_time, color_byte, color_time = [], [], [], []
        for i in range(SliceNum):
            positions_byte.append(parse_helper('positions bitstream size ', ' B', i))
            positions_time.append(parse_helper('positions processing time (user): ', ' s', i))
            color_byte.append(parse_helper(f'{attriName} bitstream size ', ' B', i))
            color_time.append(parse_helper(f'{attriName} processing time (user): ', ' s', i))
        Total_size = (parse_helper('Total bitstream size ', ' B'))
        Total_time = (parse_helper('Processing time (user): ', ' s'))
        result =  {'pos_bits':sum(positions_byte)*8,'atr_bits':sum(color_byte)*8,\
                    'total_bits':Total_size*8,'en_pos_time':np.round(sum(positions_time),5),\
                    'en_atr_time':np.round(sum(color_time),5),'en_total_time':Total_time*1.0}
        return result

    if type in ['tmc13Decode']:
        Total_time = (parse_helper('Processing time (user): ', ' s'))
        de_atr_time = (parse_helper(f'{attriName} processing time (user): ', ' s'))
        result = {'de_total_time': Total_time * 1.0, 'de_atr_time': de_atr_time * 1.0}
        return result

    if type in ['pc_error']:
        msed1 = parse_helper('mseF      (p2point): ')
        psnrd1 = parse_helper('mseF,PSNR (p2point): ')
        msed2 = parse_helper('mseF      (p2plane): ')
        psnrd2 = parse_helper('mseF,PSNR (p2plane): ')
        mseY = parse_helper('c[0],    F         : ')
        mseU = parse_helper('c[1],    F         : ')
        mseV = parse_helper('c[2],    F         : ')
        psnrY = parse_helper('c[0],PSNRF         : ')
        psnrU = parse_helper('c[1],PSNRF         : ')
        psnrV = parse_helper('c[2],PSNRF         : ')
        result = {'d1': psnrd1, 'd2': psnrd2, 'Y': psnrY, 'U': psnrU, 'V': psnrV, 'msed1d2YUV': np.c_[msed1, msed2, mseY, mseU, mseV]}
        return result


def run_cmd(cmd, print_screen=False, log_path=""):
    out = subprocess.check_output(cmd, shell=True, encoding='utf-8', errors='ignore')
    if print_screen:
        cprint(cmd, 'red', 'on_green')
        print(out, flush=True)
    if len(log_path):
        with open(log_path, 'w', newline='') as f:
            f.write(out)
    return out


class Cpt():

    def __init__(self, path, attributeType=None, className=None, create_new=False) -> None:
        if create_new is False and not os.path.exists(path):
            self = None
        self.path = path
        self.className = className
        self.source_path = None
        self.recon_path = None
        self.inputPointNum = None
        self.raw_pt = None
        self.quan_pt = None
        self.dequan_pt = None
        self.quan_parm = None
        self.quan_path = path
        self.attriType = attributeType
        if self.attriType == 'rgb': self.attriType = "colors"
        if self.attriType == 'ref': self.attriType = 'reflectances'
        assert self.attriType in ['colors', 'reflectances'], self.attriType

    def read(self, property=None):
        if property is None: property = [self.attriType]
        self.raw_pt = np.concatenate(ptIO.pcread(self.path, True), -1)
        self.inputPointNum = self.raw_pt.shape[0]
        return self.raw_pt[:, 0:3], self.raw_pt[:, 3:]

    def getInputPointNum(self):
        self.read()
        return self.inputPointNum

    def save(self, data, path):
        if self.attriType == "reflectances":
            ptIO.write_ply_data(path, data, attributeName=['reflectance'], attriType=['uint16'])
        if self.attriType == "colors":
            ptIO.write_ply_data(path, data, attributeName=['red', 'green', 'blue'], attriType=['uchar', 'uchar', 'uchar'])

    def Quantization(self, qlevel=18, offset='mean', qs=None, atq=100):
        # self.read()
        refPt = self.raw_pt[:, 0:3].copy()
        refAtr = self.raw_pt[:, 3:].copy()
        offsetPt = offset
        if offset == 'min':
            offsetPt = refPt.min(0, keepdims=True)
        if offset == 'mean':
            offsetPt = refPt.mean(0, keepdims=True)
        points = refPt - offsetPt
        if qlevel is not None:
            qs = (points.max() - points.min()) / (2**qlevel - 1)
        pt = np.round(points / qs)
        at = np.round(refAtr * atq)
        pt, idx = np.unique(pt, axis=0, return_index=True)
        at = at[idx]
        self.quan_parm = {'qs': qs, 'offset': offsetPt, 'atq': atq}
        self.quan_pt = np.concatenate((pt, at), -1)
        return self.quan_pt.astype(int)

    def deQuantization(self, quan_pt=None):
        if quan_pt is None: quan_pt = self.quan_pt
        dePt = quan_pt[:, 0:3] * self.quan_parm['qs'] + self.quan_parm['offset']
        deAt = quan_pt[:, 3:] / self.quan_parm['atq']
        self.dequan_pt = np.concatenate((dePt, deAt), -1)
        return self.dequan_pt

    def compressByTmc(self, Config, Path=None, otherParams='', BinPath='./testTool/temp/tmc/tmc.bin', print_scree=False, log_path=""):
        if Path is None: Path = self.quan_path
        if Path is None: Path = self.path
        self.source_path = Path
        os.makedirs(os.path.dirname(BinPath), exist_ok=True)
        cmd = "{} --mode=0 --uncompressedDataPath={} -c {} --compressedStreamPath={} {}".format(TMC_PATH, self.source_path, Config, BinPath, otherParams)
        out = run_cmd(cmd, print_scree, log_path)
        return parse_cmd_output(out, type='tmc13Encode', attriName=self.attriType)

    def deCompressByTmc(self, otherParams='', BinPath='./testTool/temp/tmc/tmc.bin', OutputPath='./testTool/temp/tmc/recPt.ply', print_scree=False, log_path=""):
        self.recon_path = OutputPath
        cmd = "{} --mode=1 --reconstructedDataPath={} --compressedStreamPath={} {}".format(TMC_PATH, self.recon_path, BinPath, otherParams)
        out = run_cmd(cmd, print_scree, log_path)
        return parse_cmd_output(out, type='tmc13Decode', attriName=self.attriType)

    def psnrMPEG(self, peak_value, print_scree=False, log_path=""):
        cmd = "{} -a {} -b {} -r {}".format(PCERROR_PATH, self.source_path, self.recon_path, peak_value)
        out = run_cmd(cmd, print_scree, log_path)
        return parse_cmd_output(out, type='pc_error')

    def psnrOurs(self, peak_value, print_scree=False):
        d = pc_error(self.raw_pt[:, :3], self.dequan_pt[:, :3], YUV_PSNR=False, peakvalue=peak_value, detail=print_scree)
        for key in list(d.keys()):
            if np.isnan(d[key]): d.pop(key)
        return d

    def psnrReflectance(self):
        return getRefPSNR(self.raw_pt, self.dequan_pt, peak_value=1.0)


class TestFile():

    def __init__(self, path, filter_fun=None, attributeType='rgb', classFun=None) -> None:
        self.path = []
        for f in glob.glob(path):
            if filter_fun is None or filter_fun(f):
                self.path.append(f)
        self.path = sorted(self.path, key=lambda x: x.lower())
        self.getClass_fun = classFun
        self._results = []
        self.attributeType = attributeType
        print(self.path)

    def fileIt(self):
        for path in self.path:
            className = self._getClass(path)
            yield Cpt(path, className=className, attributeType=self.attributeType)

    def _getClass(self, path):
        if self.getClass_fun is not None:
            return self.getClass_fun(path)
        else:
            return '__default__class'

    def testByTMC(self, Config='testTool/gpcc_cw.cfg', otherParams='', preProcess_fun=None, print=print):
        self._results = []
        pbar = tqdm(total=len(self.path))
        for i in self.fileIt():
            try:
                pbar.update(1)
                if preProcess_fun is not None:
                    preProcess_fun(i)
                tmc_encode = i.compressByTmc(Config, otherParams=otherParams, print_scree=False)
                infos = {'file': os.path.basename(i.path)}
                if self.getClass_fun is not None:
                    infos['class'] = i.className
                anchor_res = {'atr_bits': tmc_encode['atr_bits'], 'atr_bpip': tmc_encode['atr_bits'] / i.getInputPointNum(), 'en_atr_time': tmc_encode['en_atr_time'], 'ptNum': i.inputPointNum}
                res = dict(infos, **anchor_res)
                self._results.append(res)
            except:
                res = i.path + ' error!'
            if print is not None:
                self.printFun(res)
        table = pd.DataFrame(self._results)
        table = pd.concat([table, pd.DataFrame([table.mean(numeric_only=True)])], ignore_index=True)
        table = table.round(3)
        return table

    def preProcessFun(self, pt: Cpt):
        pt.Quantization(qlevel=None, qs=1.0, offset=0, atq=1.0)
        pt.quan_path = pt.path
        # pt.quan_path = 'Data/kittiQ/Q{}/{}/{}'.format(QLEVEL,pt.className,os.path.basename(pt.path)[:-3]+'ply')
        # pointCloud.write_ply_data(pt.quan_path ,pt.quan_pt,attributeName=['reflectance'],attriType=['uint16'])

    def printFun(self, value, print=print):
        if print is not None:
            s = str(value)
            s = re.sub(r'array|\(|\)|\{|\}|\'', '', s)
            print(re.sub(r'(?<!\w)(\d+)(\.\d{0,3})\d*(?!\w)', r'\1\2', s))

    def testByFun(self, encoding_fun, print=print, testGPCC=['enc'],lidar=False):
        self._results = []
        pbar = tqdm(total=len(self.path))
        for i in self.fileIt():
            pbar.update(1)
            # self.preProcessFun(i)
            res = {}
            encode_res = encoding_fun(i.path)
            res.update(encode_res)
            if 'enc' in testGPCC:
                if lidar:
                    anchor = i.compressByTmc(Config='testTool/gpcc_cw_lidar_angular_off.cfg',print_scree=False)
                else:
                    anchor = i.compressByTmc(Config='testTool/gpcc_cw.cfg', print_scree=False)
                res.update({'gpcc_time': anchor['en_atr_time'], 'gpcc_bpip': anchor['atr_bits'] / res['inPtNum'], 'gain': (encode_res['Bits'] - anchor['atr_bits']) / anchor['atr_bits'] * 100})
            if 'dec' in testGPCC:
                deanchor = i.deCompressByTmc(print_scree=False)
                print(deanchor)
                res.update({'gpcc_detime': deanchor['de_atr_time']})

            infos = {'File': os.path.basename(i.path)}
            if self.getClass_fun is not None:
                infos['class'] = i.className
            res.pop('Bppinfo', None)
            res = dict(infos, **res)
            self._results.append(res)
            self.printFun(res, print=print)

        table = pd.DataFrame(self._results)
        table = pd.concat([table, pd.DataFrame([table.mean(numeric_only=True)])], ignore_index=True)
        table = table.round(3)
        return table


if __name__ == '__main__':
    ## for MPEGCAT1
    # Test = TestFile('Data/MPEG/MPEGCat1A/*.ply',attributeType='rgb')
    # table = Test.testByTMC(Config='testTool/gpcc_cw.cfg')

    ## for semanticKITTI
    # def className(path):
    #     return int(path.split('sequences/')[1].split('/')[0])
    # def filter_fun(path):
    #     return className(path)>10
    # def preProcessFun(pt: Cpt, vox=16):
    #     pt.read()
    #     pt.Quantization(qlevel=None, qs=400/(2**vox-1), offset=-200, atq=100)
    #     seq = pt.className
    #     pt.quan_path = f'semanticKITTI_vox{vox}/{seq}/'+os.path.basename(pt.path)[:-3]+'ply'
    #     pointCloud.write_ply_data(pt.quan_path, pt.quan_pt, attributeName=['reflectance'], attriType=['uint16'])
    # Test = TestFile(f'Data/semanticKITTI/dataset/sequences/*/velodyne/*.bin',attributeType='ref',filter_fun=filter_fun,classFun=className)
    # table = Test.testByTMC(Config='testTool/gpcc_cw_lidar_angular_off.cfg',preProcess_fun=preProcessFun)

    ## for ford

    qs = 20

    def className(path):
        return int(path.split('Ford_')[1].split('_q')[0])

    def filter_fun(path):
        return className(path) > 1

    def preProcessFun(pt: Cpt, qs=qs):
        pt.read()
        pt.Quantization(qlevel=None, qs=qs, offset=0, atq=1)
        seq = pt.className
        pt.quan_path = f'ford_q{qs}mm/{seq}/' + os.path.basename(pt.path)[:-3] + 'ply'
        ptIO.write_ply_data(pt.quan_path, pt.quan_pt, attributeName=['reflectance'], attriType=['uint16'])

    Test = TestFile(f'Data/MPEG/MPEGCat3Frame/Ford*/*.ply', attributeType='ref', filter_fun=filter_fun, classFun=className)
    table = Test.testByTMC(Config='testTool/gpcc_cw_lidar_angular_off.cfg', preProcess_fun=preProcessFun)
    table.to_csv(f'Exp/GPCC/ford{qs}mm.csv', index=False)

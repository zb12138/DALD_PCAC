'''
Author: chunyangf@qq.com
LastEditors: chunyang fu
Description: runlength coding wrapper, call resAc.so
Date: 2024-11-29 16:59:12
All rights reserved.
'''
from ctypes import *
from tkinter import TRUE
from tkinter.messagebox import NO
import numpy as np
import os
import time

lib = cdll.LoadLibrary(os.path.dirname(os.path.abspath(__file__)) + '/resAc.so')  # class level loading lib
lib.encoding.restype = c_void_p
lib.encoding.argtypes = [POINTER(c_int32), c_int, c_int, POINTER(c_uint8), POINTER(c_uint32), c_bool]
lib.decoding.restype = c_void_p
lib.decoding.argtypes = [POINTER(c_uint8), c_int, c_int, c_int, POINTER(c_int32)]


def encode_res(data, detail=False):
    if np.ndim(data) == 1:
        channel = 1
    else:
        channel = data.shape[1]
    data = np.reshape(data.T, data.shape, order='C')
    data = np.ascontiguousarray(data).astype(np.int32)
    data_p = data.ctypes.data_as(POINTER(c_int32))
    code = (c_uint8 * (data.shape[0] * channel * 32))()
    code_len = c_uint32()
    lib.encoding(data_p, data.shape[0], channel, code, byref(code_len), detail)
    return np.array(code[0:code_len.value])


def decode_res(code, pointNum, channel):
    code = np.ascontiguousarray(code).reshape(-1).astype(np.uint8)
    code_p = code.ctypes.data_as(POINTER(c_uint8))
    data = (c_int32 * (pointNum * channel))()
    lib.decoding(code_p, code.shape[0], pointNum, channel, data)
    return np.array(data[0:pointNum * channel]).reshape(channel, pointNum).T


if __name__ == "__main__":
    data = np.random.randint(-1000, 1000, (100000, 3))
    code = encode_res(data, True)
    data_d = decode_res(code, 100000, 3)
    print((data == data_d).all())

'''
Author: chunyangf@qq.com
LastEditors: chunyang fu
Description: octree generation wrapper, call Octree_python_lib.so
Date: 2024-11-29 16:59:12
All rights reserved.
'''
from ctypes import *
from tkinter import TRUE
from tkinter.messagebox import NO
import numpy as np
import os
import time


class CNode(Structure):
    _fields_ = [
        ('nodeid', c_uint),
        ('octant', c_uint),
        ('parent', c_uint),
        ('oct', c_uint8),
        # ('pointIdx',c_void_p),
        ('childPoint', c_void_p * 8),
        ('pos', c_uint * 3)
    ]


c_double_p = POINTER(c_double)
c_uint_p = POINTER(c_uint32)
lib = cdll.LoadLibrary(os.path.dirname(os.path.abspath(__file__)) + '/Octree_python_lib.so')  # class level loading lib

lib.GenOctreeInit.restype = c_void_p
lib.GenOctreeInit.argtypes = [c_double_p, c_uint_p, c_int]
lib.genOctreeInterface.restype = c_void_p
lib.genOctreeInterface.argtypes = [c_void_p]

lib.getOctree.restype = c_void_p
lib.getOctree.argtypes = [c_void_p]

lib.delete_vector.restype = None
lib.delete_vector.argtypes = [c_void_p]
lib.vector_size.restype = c_int
lib.vector_size.argtypes = [c_void_p]
lib.vector_get.restype = c_void_p
lib.vector_get.argtypes = [c_void_p, c_int]
lib.vector_push_back.restype = None
lib.vector_push_back.argtypes = [c_void_p, c_int]

lib.Nodes_get.argtypes = [c_void_p, c_int, c_bool]
lib.Nodes_get.restype = POINTER(CNode)

lib.delete_Nodes.argtypes = [c_void_p]
lib.delete_Nodes.restype = None
# lib.Nodes_get2.argtypes = [c_void_p,c_int]
# lib.Nodes_get2.restype = POINTER(Node)

lib.Nodes_size.restype = c_int
lib.Nodes_size.argtypes = [c_void_p]

lib.int_size.restype = c_int
lib.int_size.argtypes = [c_void_p]

lib.int_get.restype = c_uint
lib.int_get.argtypes = [c_void_p, c_int]


class COctree(object):

    def __init__(self, points, color=None):
        data = np.ascontiguousarray(points).astype(np.double)
        data_p = data.ctypes.data_as(c_double_p)
        if color is not None:
            color2 = np.ascontiguousarray(color).astype(np.uint32)
            self.color_p = color2.ctypes.data_as(c_uint_p)
        else:
            self.color_p = None
        self.GenOctreeP = lib.GenOctreeInit(data_p, self.color_p, data.shape[0])  # octree pointer to new vector
        self.code = None
        self.Octree = None
        self.__octreeDepth__ = 0
        self.nodeNum = None

    def __del__(self):  # when reference count hits 0 in Python,
        lib.delete_vector(self.GenOctreeP)  # call C++ vector destructor
        # pass

    def __len__(self):
        return self.__octreeDepth__

    def __getitem__(self, i):  # access elements in vector at index
        L = self.__len__()
        if i >= L or i < -L:
            raise IndexError('Vector index out of range')
        if i < 0:
            i += L
        return Level(lib.vector_get(self.Octree, c_int(i)), i, i == L - 1)

    def __repr__(self):
        return '[{}]'.format(', '.join(str(self[i]) for i in range(len(self))))

    def genOctree(self):  # foo in Python calls foo in C++
        self.code = Vector(lib.genOctreeInterface(self.GenOctreeP))
        self.Octree = lib.getOctree(self.GenOctreeP)
        self.__octreeDepth__ = lib.vector_size(self.Octree)
        self.nodeNum = self[-1][-1].nodeid

    def GenKparentSeq(self, K):
        S1 = 6 * K * self.nodeNum
        S2 = 3 * 8 * self.nodeNum
        lib.GenKparentSeqInterface.restype = None
        lib.GenKparentSeqInterface.argtypes = [c_void_p, c_int, POINTER(POINTER(c_int)), POINTER(POINTER(c_int))]
        p = POINTER(c_int)()
        p2 = POINTER(c_int)()
        lib.GenKparentSeqInterface(self.GenOctreeP, K, p, p2)
        seqPos = np.array([p[x] for x in range(S1)]).reshape(self.nodeNum, K, 6)
        if self.color_p is not None:
            seqAttri = np.array([p2[x] for x in range(S2)]).reshape(self.nodeNum, 8, 3)
        else:
            seqAttri = None
        return seqPos, seqAttri


class Vector():

    def __init__(self, Adr) -> None:
        self.nodeAdr = Adr
        self.Len = lib.int_size(Adr)

    def __repr__(self):
        return '[{}]'.format(', '.join(str(self[i]) for i in range(len(self))))

    def __getitem__(self, i):
        L = self.Len
        if i >= L or i < -L:
            raise IndexError('Vector index out of range')
        if i < 0:
            i += L
        return lib.int_get(self.nodeAdr, i)

    def __len__(self):
        return lib.int_size(self.nodeAdr)


class Level():

    def __init__(self, Adr, i, leaf) -> None:
        self.Adr = Adr
        self.level = i + 1
        self.Len = lib.Nodes_size(self.Adr)
        self.nodeAdr = None
        self.leaf = leaf

    def __getitem__(self, i):
        L = self.Len
        if i >= L or i < -L:
            raise IndexError('Vector index out of range')
        if i < 0:
            i += L
        return Node(self.Adr, self.leaf, self.level, i)

    def __len__(self):
        print('len', lib.Nodes_size(self.Adr))
        return lib.Nodes_size(self.Adr)

    def __repr__(self):
        return '\n level {}, len {}'.format(self.level, self.Len)


class Node():

    def __init__(self, Adr, leaf, level, i) -> None:
        self.Adr = lib.Nodes_get(Adr, i, leaf)
        __content__ = self.Adr.contents
        self.oct = __content__.oct
        self.nodeid = __content__.nodeid
        self.parent = __content__.parent
        self.octant = __content__.octant
        self.pos = np.array(__content__.pos)
        if leaf:
            self.childPoint = [Vector(__content__.childPoint[x]) for x in range(8)]
        else:
            self.childPoint = [[]] * 8
        self.level = level
        self.isleaf = leaf

    def __repr__(self):
        return 'nodeid {}, oct {}, parent {}, octant {}, pos {}, level {}, isleaf {}, childPoint {}'\
            .format(self.nodeid, self.oct, self.parent, self.octant, self.pos, self.level, self.isleaf, self.childPoint)

    def __del__(self):
        lib.delete_Nodes(self.Adr)


def GenOctree(points, color=None):
    Octree = COctree(points, color)
    Octree.genOctree()
    # Octree.GenKparentSeq(4)
    return Octree.code, Octree, len(Octree)

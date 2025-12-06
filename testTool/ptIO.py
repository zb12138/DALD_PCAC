import numpy as np
import ctypes
import os 
class PCC_IO:
    def __init__(self):
        self.lib = ctypes.CDLL( os.path.join(os.path.dirname(__file__), "ptio_backend.so"))
        self._define_function_prototypes()
        self.obj = self.lib.pcc_io_create()
    
    def __del__(self):
        if self.obj:
            self.lib.pcc_io_destroy(self.obj)


    def _define_function_prototypes(self):
        self.lib.pcc_io_create.restype = ctypes.c_void_p
        
        self.lib.pcc_io_destroy.argtypes = [ctypes.c_void_p]
        
        self.lib.pcc_io_read.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double
        ]
        self.lib.pcc_io_read.restype = ctypes.c_bool
        
        self.lib.pcc_io_write.argtypes = [
            ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double, ctypes.c_bool
        ]
        self.lib.pcc_io_write.restype = ctypes.c_bool
        
        self.lib.pcc_io_get_point_count.argtypes = [ctypes.c_void_p]
        self.lib.pcc_io_get_point_count.restype = ctypes.c_size_t
        
        self.lib.pcc_io_has_colors.argtypes = [ctypes.c_void_p]
        self.lib.pcc_io_has_colors.restype = ctypes.c_bool
        
        self.lib.pcc_io_has_reflectance.argtypes = [ctypes.c_void_p]
        self.lib.pcc_io_has_reflectance.restype = ctypes.c_bool
        
        self.lib.pcc_io_get_points.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.pcc_io_get_points.restype = ctypes.c_bool
        
        self.lib.pcc_io_get_colors.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.pcc_io_get_colors.restype = ctypes.c_bool
        
        self.lib.pcc_io_get_reflectance.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)),
            ctypes.POINTER(ctypes.c_size_t)
        ]
        self.lib.pcc_io_get_reflectance.restype = ctypes.c_bool
        
        self.lib.pcc_io_set_points.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_double), 
            ctypes.c_size_t
        ]
        self.lib.pcc_io_set_points.restype = ctypes.c_bool
        
        self.lib.pcc_io_set_colors.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_uint8), 
            ctypes.c_size_t
        ]
        self.lib.pcc_io_set_colors.restype = ctypes.c_bool
        
        self.lib.pcc_io_set_reflectance.argtypes = [
            ctypes.c_void_p, 
            ctypes.POINTER(ctypes.c_uint16), 
            ctypes.c_size_t
        ]
        self.lib.pcc_io_set_reflectance.restype = ctypes.c_bool
        
        self.lib.pcc_io_clear.argtypes = [ctypes.c_void_p]
        self.lib.pcc_io_remove_colors.argtypes = [ctypes.c_void_p]
        self.lib.pcc_io_remove_reflectance.argtypes = [ctypes.c_void_p]

        self.lib.pcc_io_free_double_array.argtypes = [ctypes.POINTER(ctypes.c_double)]
        self.lib.pcc_io_free_uint8_array.argtypes = [ctypes.POINTER(ctypes.c_uint8)]
        self.lib.pcc_io_free_uint16_array.argtypes = [ctypes.POINTER(ctypes.c_uint16)]

    def read(self, filename, position_scale=1.0):
        filename_bytes = filename.encode('utf-8')
        return self.lib.pcc_io_read(self.obj, filename_bytes, position_scale)
    
    def write(self, filename, position_scale=1.0, as_ascii=False):
        filename_bytes = filename.encode('utf-8')
        return self.lib.pcc_io_write(self.obj, filename_bytes, position_scale, as_ascii)
    
    def get_point_count(self):
        return self.lib.pcc_io_get_point_count(self.obj)
    
    def has_colors(self):
        return self.lib.pcc_io_has_colors(self.obj)
    
    def has_reflectance(self):
        return self.lib.pcc_io_has_reflectance(self.obj)
    
    def get_points(self):
        point_count = ctypes.c_size_t(0)
        points_ptr = ctypes.POINTER(ctypes.c_double)()
        success = self.lib.pcc_io_get_points(
            self.obj, ctypes.byref(points_ptr), ctypes.byref(point_count)
        )
        if not success or point_count.value == 0:
            return np.zeros((0, 3), dtype=np.float64)
        np_array = np.ctypeslib.as_array(points_ptr, shape=(point_count.value * 3,))
        np_array = np_array.reshape((point_count.value, 3))
        result = np.copy(np_array)
        self.lib.pcc_io_free_double_array(points_ptr)
        return result
    
    def get_colors(self):
        if not self.has_colors():
            return np.zeros((0, 3), dtype=np.uint8)
        
        point_count = ctypes.c_size_t(0)
        colors_ptr = ctypes.POINTER(ctypes.c_uint8)()
        
        success = self.lib.pcc_io_get_colors(
            self.obj, ctypes.byref(colors_ptr), ctypes.byref(point_count)
        )
        
        if not success or point_count.value == 0:
            return np.zeros((0, 3), dtype=np.uint8)

        np_array = np.ctypeslib.as_array(colors_ptr, shape=(point_count.value * 3,))
        np_array = np_array.reshape((point_count.value, 3))
        result = np.copy(np_array)
        self.lib.pcc_io_free_uint8_array(colors_ptr)
        return result
    
    def get_reflectance(self):
        if not self.has_reflectance():
            return np.zeros(0, dtype=np.uint16)
        point_count = ctypes.c_size_t(0)
        reflectance_ptr = ctypes.POINTER(ctypes.c_uint16)()
        success = self.lib.pcc_io_get_reflectance(
            self.obj, ctypes.byref(reflectance_ptr), ctypes.byref(point_count)
        )
        if not success or point_count.value == 0:
            return np.zeros(0, dtype=np.uint16)
        np_array = np.ctypeslib.as_array(reflectance_ptr, shape=(point_count.value,1))
        result = np.copy(np_array)
        self.lib.pcc_io_free_uint16_array(reflectance_ptr)
        return result
    
    def set_points(self, points):
        if not isinstance(points, np.ndarray):
            points = np.array(points, dtype=np.float64)
        
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Points must be a 2D array with shape (N, 3)")
        point_count = points.shape[0]
        points_flat = points.flatten().astype(np.float64)
        points_ptr = points_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        return self.lib.pcc_io_set_points(self.obj, points_ptr, point_count)
    
    def set_colors(self, colors):
        if not isinstance(colors, np.ndarray):
            colors = np.array(colors, dtype=np.uint8)
        if colors.ndim != 2 or colors.shape[1] != 3:
            raise ValueError("Colors must be a (N, 3) array")
        point_count = colors.shape[0]
        colors_flat = colors.flatten().astype(np.uint8)
        colors_ptr = colors_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        return self.lib.pcc_io_set_colors(self.obj, colors_ptr, point_count)
    
    def set_reflectance(self, reflectance):
        if not isinstance(reflectance, np.ndarray):
            reflectance = np.array(reflectance, dtype=np.uint16)
        if reflectance.ndim != 2 or reflectance.shape[1] != 1:
            raise ValueError("Reflectance must be a (N, 1) array")
        point_count = reflectance.size
        reflectance_ptr = reflectance.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))
        return self.lib.pcc_io_set_reflectance(self.obj, reflectance_ptr, point_count)
    
    def clear(self):
        self.lib.pcc_io_clear(self.obj)
    
    def remove_colors(self):
        self.lib.pcc_io_remove_colors(self.obj)
    
    def remove_reflectance(self):
        self.lib.pcc_io_remove_reflectance(self.obj)


def pcwrite(file_path: str, points: np.ndarray, attribute:np.ndarray = None, colors: np.ndarray = None, reflectance: np.ndarray = None, asAscii: bool = False):
    """
    Write point cloud data to a file.
    :param file_path: Path to the output point cloud file.
    :param points: NumPy array of points (shape: [N, 3]).
    :param attribute: Optional NumPy array of attributes (shape: [N, 3] or [N, 1]).
    :param colors: Optional NumPy array of colors (shape: [N, 3]).
    :param reflectance: Optional NumPy array of reflectance values (shape: [N, 1]).
    :param asAscii: Whether to write the file in ASCII format.
    """
    pc = PointCloud()
    pc.set_points(points)
    if attribute is not None:
        if attribute.shape[1] == 1:
            reflectance = attribute.astype(np.uint16)
        elif attribute.shape[1] == 3:
            colors = attribute.astype(np.uint8)
        else:
            raise ValueError("Attribute must be a 1D array or a 2D array with shape (N, 3).")
    if colors is not None:
        pc.set_colors(colors)
    if reflectance is not None:
        pc.set_reflectance(reflectance)
    if os.path.dirname(file_path)!='': os.makedirs(os.path.dirname(file_path), exist_ok=True)
    pc.write(file_path, asAscii)
    return pc 

def pcread(file_path: str, attribute=True, return_pc: bool = False) -> tuple:
    """
    Read point cloud data from a file.
    :param file_path: Path to the point cloud file.
    :param attribute: Whether to read color/reflectance.
    :param return_pc: If True, return the PointCloud object.
    :return: A tuple containing points, colors, and reflectance (if requested).
    """
    pc = PointCloud()
    pc.read(file_path, attribute, attribute)
    if return_pc:
        return pc
    if attribute and pc.has_colors():
        return pc.points, pc.colors
    if attribute and pc.has_reflectance():
        return pc.points, pc.reflectance
    if attribute:
        return pc.points, np.empty((0, 0))
    return pc.points

class PointCloud:
    def __init__(self):
        self.points = np.empty((0, 3), dtype=np.float64)
        self.colors = np.empty((0, 3), dtype=np.uint8)
        self.reflectance = np.empty((0, 1), dtype=np.uint16)
        self.path = None 
        self._pc = PCC_IO()
        self.positionScale = 1.0 # save the point cloud in *positionScale to avoid precision loss

    def read(self, file_path, colors=False, reflectance=False):
        self.clear()
        self._pc.read(file_path,1/self.positionScale)   
        self.path = file_path
        self.read_points()
        if colors:
            self.read_colors()
        if reflectance:
            self.read_reflectance()
        if colors:
            return self.points, self.colors
        elif reflectance:
            return self.points, self.reflectance
        else:
            return self.points

    def write(self, file_path, asAscii=False):
        self._pc.write(file_path, self.positionScale, asAscii)

    def read_points(self):
        self.points = self._pc.get_points() 
        return self.points

    def read_colors(self):
        if self._pc.has_colors():
            self.colors = self._pc.get_colors() 
        return self.colors

    def read_reflectance(self):
        if self._pc.has_reflectance():
            self.reflectance = self._pc.get_reflectance() 
        return self.reflectance

    def clear(self):
        self._pc.clear()
        self._pc.remove_colors()
        self._pc.remove_reflectance()

    def set_points(self, points):
        points = (points).astype(np.float64)
        points = np.ascontiguousarray(points)
        self._pc.set_points(points)

    def set_reflectance(self, reflectance):
        reflectance = reflectance.astype(np.uint16)
        reflectance = np.ascontiguousarray(reflectance)
        self._pc.set_reflectance(reflectance)

    def set_colors(self, colors):
        colors = colors.astype(np.uint8)
        colors = np.ascontiguousarray(colors)
        self._pc.set_colors(colors)

    def has_colors(self):
        return self._pc.has_colors()

    def has_reflectance(self):
        return self._pc.has_reflectance()

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

if __name__=='__main__':
    file = "longdress_vox10_1051.ply"
    p,c = pcread(file)
    pcwrite("test.ply",p,c,asAscii=True)
    import open3d as o3d
    pc_o3d = o3d.io.read_point_cloud("test.ply")
    c2 = np.asarray(pc_o3d.colors)*255
    assert (c == c2).all()

    pc_o3d = o3d.io.read_point_cloud(file)
    c3 = np.asarray(pc_o3d.colors)*255
    assert (c == c3).all()

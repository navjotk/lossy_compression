from ctypes import *
import numpy as np


# Load the share library
mkl = cdll.LoadLibrary("libmkl_rt.dylib")

cblas_scopy = mkl.cblas_scopy
c_float_p = POINTER(c_float)

def np_ref_address(data):
    return data.ctypes.data_as(c_float_p)


def bcopy(src, dest):
    length = src.size * int((src.itemsize/4))
    cblas_scopy(c_int(length), np_ref_address(src), c_int(1), np_ref_address(dest), c_int(1))


a = np.arange(1, 100, 2, dtype=np.float32)

b = np.empty_like(a)

bcopy(a, b)
assert(np.allclose(a, b))



a = np.arange(1, 100, 2, dtype=np.float64)

b = np.empty_like(a)

bcopy(a, b)
assert(np.allclose(a, b))

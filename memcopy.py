from ctypes import *
import numpy as np
from contexttimer import Timer


# Load the share library
mkl = cdll.LoadLibrary("libmkl_rt.so")

cblas_scopy = mkl.cblas_scopy
c_float_p = POINTER(c_float)

def np_ref_address(data):
    return data.ctypes.data_as(c_float_p)


def bcopy(src, dest):
    length = src.size * int((src.itemsize/4))
    cblas_scopy(c_int(length), np_ref_address(src), c_int(1), np_ref_address(dest), c_int(1))

def numpy_copy(src, dest):
    src[:] = dest[:]

def numpy_copyto(src, dest):
    np.copyto(dest, src)

def ctype_copy(src, dest):
    length = src.size * src.itemsize
    memmove(np_ref_address(dest), np_ref_address(src), length)


functions_to_measure = {'blas': bcopy, 'numpy': numpy_copy, 'ctypes': ctype_copy, 'numpy_copyto': numpy_copyto}#,'stream': stream_copy,
                                          #'devito': devito_copy}

import time
with Timer(factor=1000) as t0:
    time.sleep(1)
print(t0.elapsed)

for name, fn in functions_to_measure.items():
    a = np.random.rand(1000000,dtype=np.float32)
    b = np.empty_like(a)
    with Timer(factor=1000) as t1:
        fn(a, b)
    assert(np.allclose(a, b))
    size_sp = a.size * a.itemsize

    a = np.arange(1, 1000000, 2, dtype=np.float64)
    b = np.empty_like(a)
    with Timer(factor=1000) as t2:
        fn(a, b)
    assert(np.allclose(a, b))
    size_dp = a.size * a.itemsize
    print(t1.elapsed, t2.elapsed)
    print(size_sp/(t1.elapsed), size_dp/(t2.elapsed))
    print("Copy method: %s, bandwidth for double precision: %f, single precision: %f" % (name, size_sp/(t1.elapsed*1000), size_dp/(t2.elapsed*1000)))

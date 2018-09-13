from ctypes import cdll
import numpy as np
import ctypes

class BitStream(ctypes.Structure):
    pass

class ZfpExecution(ctypes.Structure):
    _fields_ = [("policy", ctypes.c_int), ("params", ctypes.c_void_p)]

    
class ZfpStream (ctypes.Structure):
    _fields_ = [ ("stream", ctypes.POINTER(BitStream)), ("minbits", ctypes.c_int),
                 ("maxbits", ctypes.c_int), ("maxprec", ctypes.c_double),
                 ("minexp", ctypes.c_double), ("exec", ZfpExecution)]

class ZfpField(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("nx", ctypes.c_int), ("ny", ctypes.c_int),
                ("nz", ctypes.c_int), ("sx", ctypes.c_int), ("sy", ctypes.c_int),
                ("sz", ctypes.c_int), ("data", ctypes.c_void_p)]



libzfp = cdll.LoadLibrary("libzfp.so")
libzfp.zfp_stream_set_accuracy.argtypes = (ctypes.POINTER(ZfpStream), ctypes.c_double)
libzfp.zfp_stream_set_precision.argtypes = (ctypes.POINTER(ZfpStream), ctypes.c_int)
libzfp.zfp_stream_maximum_size.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
libzfp.zfp_stream_set_bit_stream.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
libzfp.zfp_stream_rewind.argtypes = (ctypes.c_void_p,)
libzfp.zfp_stream_open.restype = ctypes.POINTER(ZfpStream)
libzfp.zfp_field_1d.restype = ctypes.POINTER(ZfpField)
libzfp.zfp_field_2d.restype = ctypes.POINTER(ZfpField)
libzfp.zfp_field_3d.restype = ctypes.POINTER(ZfpField)
libzfp.stream_open.restype = ctypes.POINTER(BitStream)
#libzfp.zfp_stream_open.restype = ctypes.c_void_p



def raw_pointer(numpy_array):
    #return numpy_array.__array_interface__['data'][0]
    return ctypes.c_void_p(numpy_array.ctypes.data)

def compress(indata, tolerance=None, precision=None):
    print("Shape: %s" % str(indata.shape))
    assert(tolerance or precision)
    assert(not(tolerance is not None and precision is not None))
    zfp_types = {np.dtype('float32'): 3, np.dtype('float64'): 4}
    zfp_fields = {1: libzfp.zfp_field_1d, 2: libzfp.zfp_field_2d, 3: libzfp.zfp_field_3d}
    data_type = zfp_types[indata.dtype]
    status = 1
    shape = indata.shape
    
    field = zfp_fields[len(shape)](raw_pointer(indata), data_type, *shape)
    from IPython import embed
    embed()
    stream = libzfp.zfp_stream_open(None)
    stream = ctypes.cast(stream, ctypes.POINTER(ZfpStream))
    field = ctypes.cast(field, ctypes.POINTER(ZfpField))
    if tolerance is not None:
        libzfp.zfp_stream_set_accuracy(stream, tolerance)
    elif precision is not None:
        libzfp.zfp_stream_set_precision(stream, precision)
    bufsize = libzfp.zfp_stream_maximum_size(stream, field)
    buff = ctypes.create_string_buffer(bufsize)
    bitstream = libzfp.stream_open(buff, bufsize)
    libzfp.zfp_stream_set_bit_stream(stream, bitstream)
    libzfp.zfp_stream_rewind(stream)
    zfpsize = libzfp.zfp_compress(stream, field)
    
    libzfp.zfp_field_free(field)
    libzfp.zfp_stream_close(stream)
    libzfp.stream_close(bitstream)
    return buff[:zfpsize]

def decompress(compressed, shape, dtype, tolerance=None, precision=None):
    print("Shape: %s" % str(shape))
    assert(tolerance or precision)
    assert(not(tolerance is not None and precision is not None))
    outdata = np.zeros(shape, dtype=dtype)
    zfp_types = {np.float32: 3, np.dtype('float64'): 4}
    zfp_fields = {1: libzfp.zfp_field_1d, 2: libzfp.zfp_field_2d, 3: libzfp.zfp_field_3d}
    data_type = zfp_types[dtype]
    field = zfp_fields[len(shape)](raw_pointer(outdata), data_type, *shape)
    stream = libzfp.zfp_stream_open(None)
    stream = ctypes.cast(stream, ctypes.POINTER(ZfpStream))
    field = ctypes.cast(field, ctypes.POINTER(ZfpField))
    if tolerance is not None:
        libzfp.zfp_stream_set_accuracy(stream, tolerance)
    elif precision is not None:
        libzfp.zfp_stream_set_precision(stream, precision)
    bufsize = libzfp.zfp_stream_maximum_size(stream, field)
    #buff = ctypes.create_string_buffer(bufsize)
    bitstream = libzfp.stream_open(compressed, bufsize)
    libzfp.zfp_stream_set_bit_stream(stream, bitstream)
    libzfp.zfp_stream_rewind(stream)
    zfpsize = libzfp.zfp_decompress(stream, field)
    
    libzfp.zfp_field_free(field)
    libzfp.zfp_stream_close(stream)
    libzfp.stream_close(bitstream)
    return outdata


#a = np.linspace(0, 100, num=10000).reshape((100, 100))
#tolerance = 0.00001
#status, compressed, size = compress(a, tolerance)

#recovered = decompress(compressed, a.shape, a.dtype, tolerance)
#print(len(a.tostring()))
#print(size)
#print(len(compressed.value))
#print(np.linalg.norm(recovered-a))
#print(recovered)

import blosc
import sys
import h5py
import numpy as np
from bloscpack import pack_ndarray_to_bytes, unpack_ndarray_from_bytes, BloscArgs
from itertools import product
from contexttimer import Timer


chunk_size_options = [int(2**i) for i in np.arange(19, 23.5, 0.5)]
shuffle_options = [blosc.SHUFFLE, blosc.BITSHUFFLE, blosc.NOSHUFFLE]
cname_options = blosc.cnames
clevel_options = list(range(10))


def compress(arr, chunk_size, shuffle, cname, clevel):
    if arr.dtype == np.float32:
        typesize = 4
    elif arr.dtype == np.float64:
        typesize == 8
    else:
        assert(False)

    return pack_ndarray_to_bytes(arr, blosc_args=BloscArgs(shuffle=shuffle,
                                                           typesize=typesize,
                                                           cname=cname,
                                                           clevel=clevel), chunk_size=chunk_size)


def decompress(compressed):
    return unpack_ndarray_from_bytes(compressed)

print(sys.argv)
filename = sys.argv[1]
f = h5py.File(filename, 'r')
input_array = f['data'][()].astype(np.dtype('float32'))

combinations = product(chunk_size_options, shuffle_options, cname_options, clevel_options)

for op_set in combinations:
    try:
        with Timer(factor=1000) as compress_time:
            compressed = compress(input_array, *op_set)

        compression_factor = (input_array.size * input_array.itemsize) / len(compressed)

        with Timer(factor=1000) as decompress_time:
            decompressed = decompress(compressed)

        assert(np.allclose(input_array, decompressed))
        print("%f, %d, %s, %d, %f, %f, %f" % (op_set[0], op_set[1], op_set[2], op_set[3], compress_time.elapsed, decompress_time.elapsed, compression_factor))
    except:
        print("Error")

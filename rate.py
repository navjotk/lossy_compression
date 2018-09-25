import h5py
import sys
from util import to_hdf5
import numpy as np
from contexttimer import Timer
from zfp import compress, decompress


filename = sys.argv[1]

f = h5py.File(filename, 'r')
uncompressed = f['data'][()].astype(np.dtype('float32'))
print("\"Size of compressed field\", \"Compression Factor\", \"Compression time\", \"Decompression time\", \"Rate\", \"Error norm\", \"Maximum error\"")
for p_i in range(1, 10):
    rate = p_i
    with Timer(factor=1000) as t:
        compressed = compress(uncompressed, rate=rate)
    
    with Timer(factor=1000) as t2:
        decompressed = decompress(compressed, uncompressed.shape, uncompressed.dtype, rate=rate)

    to_hdf5(decompressed, "decompressed-r-%d.h5"%p_i)
    error_matrix = decompressed-uncompressed
    print("%f, %f, %f, %f, %f, %f, %f" % (len(compressed), len(uncompressed.tostring())/float(len(compressed)), t.elapsed, t2.elapsed, rate, np.linalg.norm(error_matrix), np.max(error_matrix)))

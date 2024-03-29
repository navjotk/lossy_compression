import h5py
import sys
from util import to_hdf5
import numpy as np
from contexttimer import Timer
from zfp import compress, decompress
from argparse import ArgumentParser




description = ("Test for fixed precision mode of zfp")
parser = ArgumentParser(description=description)
parser.add_argument("filename", type=str, help="Filename")
p_parser = parser.add_mutually_exclusive_group(required=False)
p_parser.add_argument('--parallel', dest='parallel', action='store_true')
p_parser.add_argument('--no-parallel', dest='parallel', action='store_false')
parser.set_defaults(parallel=True)
args = parser.parse_args()

filename = args.filename
parallel = args.parallel

f = h5py.File(filename, 'r')
uncompressed = f['data'][()].astype(np.dtype('float32'))
print("\"Size of compressed field\", \"Compression Factor\", \"Compression time\", \"Decompression time\", \"Precision\", \"Error norm\", \"Maximum error\"")
for p_i in range(6, 20):
    precision = p_i
    with Timer(factor=1000) as t:
        compressed = compress(uncompressed, precision=precision, parallel=parallel)
    
    with Timer(factor=1000) as t2:
        decompressed = decompress(compressed, uncompressed.shape, uncompressed.dtype, precision=precision, parallel=parallel)

    to_hdf5(decompressed, "decompressed-p-%d.h5"%p_i)
    error_matrix = decompressed-uncompressed
    print("%f, %f, %f, %f, %f, %f, %f" % (len(compressed), len(uncompressed.tostring())/float(len(compressed)), t.elapsed, t2.elapsed, precision, np.linalg.norm(error_matrix), np.max(error_matrix)))

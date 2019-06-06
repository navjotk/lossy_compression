import h5py
import sys
from util import to_hdf5
import numpy as np
from contexttimer import Timer
from argparse import ArgumentParser




description = ("Test for fixed accuracy mode of different compressors")
parser = ArgumentParser(description=description)
parser.add_argument("filename", type=str, help="Filename")
parser.add_argument("compressor", choices=["zfp", "sz"], default="zfp")
p_parser = parser.add_mutually_exclusive_group(required=False)
p_parser.add_argument('--parallel', dest='parallel', action='store_true')
p_parser.add_argument('--no-parallel', dest='parallel', action='store_false')
parser.set_defaults(parallel=True)
args = parser.parse_args()

filename = args.filename
parallel = args.parallel
compressor = args.compressor

if compressor == "zfp":
    from pyzfp import compress, decompress
else:
    from pysz import compress, decompress

f = h5py.File(filename, 'r')
uncompressed = f['data'][()].astype(np.dtype('float64'))
print("\"Size of compressed field\", \"Compression Factor\", \"Compression time\", \"Decompression time\", \"Tolerance\", \"Error norm\", \"Maximum error\"")
for p_i in range(0, 16):
    tolerance = 0.1**p_i
    with Timer(factor=1000) as t:
        if compressor=="zfp":
            kwargs = {'parallel':parallel, 'tolerance': tolerance}
        else:
            kwargs = {'tolerance': tolerance}
        compressed = compress(uncompressed, **kwargs)
    
    with Timer(factor=1000) as t2:
        if compressor=="zfp":
            kwargs = {'parallel': parallel, 'tolerance': tolerance}
        else:
            kwargs = {}
        decompressed = decompress(compressed, uncompressed.shape, uncompressed.dtype, **kwargs)

    #to_hdf5(decompressed, "decompressed-t-%d.h5"%p_i)
    error_matrix = decompressed-uncompressed
    if p_i in (0, 8, 16):
        to_hdf5(error_matrix, "error_field-%s-%d.h5"%(compressor, p_i))
    print("%f, %f, %f, %f, %.16f, %f, %f" % (len(compressed), len(uncompressed.tostring())/float(len(compressed)), t.elapsed, t2.elapsed, tolerance, np.linalg.norm(error_matrix), np.max(error_matrix)))

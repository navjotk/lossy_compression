from argparse import ArgumentParser
from contexttimer import Timer

import numpy as np

from simple import overthrust_setup, overthrust_setup_tti
from zfp import compress, decompress


def run(tn=4000, space_order=4, kernel='OT4', nbpml=40, tolerance=0.01, parallel_compression=True, filename='', **kwargs):
    if kernel in ['OT2', 'OT4']:
        solver = overthrust_setup(filename=filename, tn=tn, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)
    elif kernel == 'TTI':
        solver = overthrust_setup_tti(filename=filename, tn=tn, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)
    else:
        raise ValueError()

    total_timesteps = solver.source.time_range.num
    u = None
    rec = None
    results = []
    for t in range(total_timesteps):
        return_values = solver.forward(u=u, rec=rec, time_m=t, time_M=t, save=False)
        rec = return_values[0]
        last_time_step = rec.shape[0] - 1
        u = return_values[1]
        uncompressed = u.data[t+1]
        with Timer(factor=1000) as time1:
            compressed = compress(uncompressed, tolerance=tolerance, parallel=parallel_compression)
        result = (t, len(uncompressed.tostring())/float(len(compressed)), time1.elapsed)
        results.append(result)
        print(result)
    with open('results.csv') as csvfile:
        writer = csv.writer(csvfile)
        for row in results:
            writer.writerow(row)
    


if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4', 'TTI'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DLE) mode")
    parser.add_argument("-n", default=4000, type=int,
                        help="Simulation Time (ms)")
    args = parser.parse_args()

    run(nbpml=args.nbpml, tn=args.n,
        space_order=args.space_order, kernel=args.kernel,
        dse=args.dse, dle=args.dle, filename='overthrust_3D_initial_model.h5')

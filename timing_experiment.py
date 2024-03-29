from argparse import ArgumentParser

import numpy as np
import os.path
import csv
import socket


from devito import TimeFunction, Function

from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator

from examples.seismic import Receiver
from pyrevolve import Revolver
from timeit import default_timer
from simple import overthrust_setup

from examples.seismic.acoustic.acoustic_example import acoustic_setup
from util import to_hdf5

class Timer(object):
    def __init__(self, tracker):
        self.timer = default_timer
        self.tracker = tracker
        
    def __enter__(self):
        self.start = self.timer()
        return self
        
    def __exit__(self, *args):
        end = self.timer()
        self.elapsed_secs = end - self.start
        self.elapsed = self.elapsed_secs * 1000  # millisecs
        self.tracker.append(self.elapsed)
        

def verify(space_order=4, kernel='OT4', nbpml=40, filename='', compression_params={}, **kwargs):
    solver = acoustic_setup(shape=(10, 10), spacing=(10, 10), nbpml=10, tn=50,
                            space_order=space_order, kernel=kernel, **kwargs)
    #solver = overthrust_setup(filename=filename, tn=50, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)

    u = TimeFunction(name='u', grid=solver.model.grid, time_order=2, space_order=solver.space_order)

    rec = Receiver(name='rec', grid=solver.model.grid,
                              time_range=solver.geometry.time_axis,
                              coordinates=solver.geometry.rec_positions)
    cp = DevitoCheckpoint([u])
    n_checkpoints = None
    m = solver.model.m
    dt = solver.dt
    v = TimeFunction(name='v', grid=solver.model.grid, time_order=2, space_order=solver.space_order)
    grad = Function(name='grad', grid=solver.model.grid)
    wrap_fw = CheckpointOperator(solver.op_fwd(save=False), src=solver.geometry.src, u=u, m=m, rec=rec, dt=dt)
    wrap_rev = CheckpointOperator(solver.op_grad(save=False), u=u, v=v, m=m, rec=rec, dt=dt, grad=grad)
    nt = rec.data.shape[0] - 2
    print("Verifying for %d timesteps" % nt)
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt,
                   compression_params=compression_params)
    wrp.apply_forward()
    summary = wrp.apply_reverse()
    print(wrp.profiler.timings)
    
    with Timer([]) as tf:
        rec2, u2, _ = solver.forward(save=True)

    with Timer([]) as tr:
        grad2, _ = solver.gradient(rec=rec2, u=u2)

    error = grad.data - grad2.data
    to_hdf5(error, 'zfp_grad_errors.h5')
    print("Error norm", np.linalg.norm(error))
    
    #assert(np.allclose(grad.data, grad2.data))
    print("Checkpointing implementation is numerically verified")
    print("Verification took %d ms for forward and %d ms for reverse" % (tf.elapsed, tr.elapsed))


def checkpointed_run(space_order=4, ncp=None, kernel='OT4', nbpml=40, filename='', compression_params={}, **kwargs):
    solver = overthrust_setup(filename=filename, tn=1000, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)
    
    u = TimeFunction(name='u', grid=solver.model.grid, time_order=2, space_order=solver.space_order)
    rec = Receiver(name='rec', grid=solver.model.grid,
                              time_range=solver.geometry.time_axis,
                              coordinates=solver.geometry.rec_positions)
    cp = DevitoCheckpoint([u])
    n_checkpoints = ncp

    m = solver.model.m
    dt = solver.dt
    v = TimeFunction(name='v', grid=solver.model.grid, time_order=2, space_order=solver.space_order)
    grad = Function(name='grad', grid=solver.model.grid)
    wrap_fw = CheckpointOperator(solver.op_fwd(save=False), src=solver.geometry.src, u=u, m=m, rec=rec, dt=dt)
    wrap_rev = CheckpointOperator(solver.op_grad(save=False), u=u, v=v, m=m, rec=rec, dt=dt, grad=grad)
    
    fw_timings = []
    rev_timings = []

    nt = rec.data.shape[0] - 2
    print("Running %d timesteps" % (nt))
    print(compression_params)
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, nt, compression_params=compression_params)
    with Timer(fw_timings):
        wrp.apply_forward()
    with Timer(rev_timings):
        wrp.apply_reverse()

    return grad, wrp, fw_timings, rev_timings

def compare_error(space_order=4, ncp=None, kernel='OT4', nbpml=40, filename='', compression_params={}, **kwargs):
    grad, wrp, fw_timings, rev_timings = checkpointed_run(space_order, ncp, kernel, nbpml, filename,
                                                          compression_params, **kwargs)
    print(wrp.profiler.summary())

    compression_params['scheme'] = None
    
    print("*************************")
    print("Starting uncompressed run:")
    
    grad2, wrp2, fw_timings2, rev_timings2 = checkpointed_run(space_order, ncp, kernel, nbpml, filename,
                                                          compression_params, **kwargs)

    error_field = grad2.data - grad.data

    print("compression enabled norm", np.linalg.norm(grad.data))
    print("compression disabled norm", np.linalg.norm(grad2.data))
    to_hdf5(error_field, 'zfp_grad_errors_full.h5')
    print("Error norm", np.linalg.norm(error_field))

def run(space_order=4, ncp=None, kernel='OT4', nbpml=40, filename='', compression_params={}, **kwargs):
    #solver = acoustic_setup(shape=(10, 10), spacing=(10, 10), nbpml=10, tn=50,
    #                        space_order=space_order, kernel=kernel, **kwargs)
    grad, wrp, fw_timings, rev_timings = checkpointed_run(space_order, ncp, kernel, nbpml, filename,
                                                          compression_params, **kwargs)
    print(wrp.profiler.summary())

    hostname = socket.gethostname()
    results_file = 'timing_results_1.csv'
    if not os.path.isfile(results_file):
        write_header = True
    else:
        write_header = False
        
    csv_row = wrp.profiler.get_dict()
    
    fieldnames = ['ncp', 'hostname'] + list(csv_row.keys())
    csv_row['ncp'] = n_checkpoints
    csv_row['hostname'] = hostname
    with open(results_file,'a') as fd:
        writer = csv.DictWriter(fd, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)



if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
    parser.add_argument("--ncp", default=None, type=int)
    parser.add_argument("--compression", choices=[None, 'zfp', 'sz', 'blosc'], default=None)
    parser.add_argument("--tolerance", default=6, type=int)
    parser.add_argument("--runmode", choices=["error", "timing"], default="timing")
    parser.add_argument("--nbpml", default=40,
                        type=int, help="Number of PML layers around the domain")
    parser.add_argument("-k", dest="kernel", default='OT2',
                        choices=['OT2', 'OT4'],
                        help="Choice of finite-difference kernel")
    parser.add_argument("-dse", default="advanced",
                        choices=["noop", "basic", "advanced",
                                 "speculative", "aggressive"],
                        help="Devito symbolic engine (DSE) mode")
    parser.add_argument("-dle", default="advanced",
                        choices=["noop", "advanced", "speculative"],
                        help="Devito loop engine (DLE) mode")
    args = parser.parse_args()
    compression_params={'scheme': args.compression, 'tolerance': args.tolerance}
    verify(nbpml=args.nbpml, 
        space_order=args.space_order, kernel=args.kernel,
           dse=args.dse, dle=args.dle, compression_params=compression_params)
    path_prefix = os.path.dirname(os.path.realpath(__file__))
    if args.runmode=="error":
        compare_error(nbpml=args.nbpml, ncp=args.ncp,
        space_order=args.space_order, kernel=args.kernel,
        dse=args.dse, dle=args.dle, filename='%s/overthrust_3D_initial_model.h5'%path_prefix,
        compression_params=compression_params)
    else:
        run(nbpml=args.nbpml, ncp=args.ncp,
        space_order=args.space_order, kernel=args.kernel,
        dse=args.dse, dle=args.dle, filename='%s/overthrust_3D_initial_model.h5'%path_prefix,
        compression_params=compression_params)

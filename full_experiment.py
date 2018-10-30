from argparse import ArgumentParser

import numpy as np

from devito import TimeFunction, Function

from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator

from examples.seismic import Receiver
from pyrevolve import Revolver

from simple import overthrust_setup

from examples.seismic.acoustic.acoustic_example import acoustic_setup
from contexttimer import Timer

def verify(space_order=4, kernel='OT4', nbpml=40, filename='', **kwargs):
    solver = acoustic_setup(shape=(10, 10), spacing=(10, 10), nbpml=10, tn=50,
                            space_order=space_order, kernel=kernel, **kwargs)
    #solver = overthrust_setup(filename=filename, tn=50, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)
    
    u = TimeFunction(name='u', grid=solver.model.grid, time_order=2, space_order=solver.space_order)
    rec = Receiver(name='rec', grid=solver.model.grid,
                              time_range=solver.receiver.time_range,
                              coordinates=solver.receiver.coordinates.data)
    cp = DevitoCheckpoint([u])
    n_checkpoints = None
    m = solver.model.m
    dt = solver.dt
    v = TimeFunction(name='v', grid=solver.model.grid, time_order=2, space_order=solver.space_order)
    grad = Function(name='grad', grid=solver.model.grid)
    wrap_fw = CheckpointOperator(solver.op_fwd(save=False), src=solver.source, u=u, m=m, rec=rec, dt=dt)
    wrap_rev = CheckpointOperator(solver.op_grad(save=False), u=u, v=v, m=m, rec=rec, dt=dt, grad=grad)

    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2)
    wrp.apply_forward()
    summary = wrp.apply_reverse()

    with Timer(factor=1000) as tf:
        rec2, u2, _ = solver.forward(save=True)

    with Timer(factor=1000) as tr:
        grad2, _ = solver.gradient(rec=rec2, u=u2)

    assert(np.allclose(grad.data, grad2.data))
    print("Checkpointing implementation is numerically verified")
    print("Verification took %d ms for forward and %d ms for reverse" % (tf.elapsed, tr.elapsed))
    

def run(space_order=4, kernel='OT4', nbpml=40, filename='', **kwargs):
    solver = acoustic_setup(shape=(10, 10), spacing=(10, 10), nbpml=10, tn=50,
                            space_order=space_order, kernel=kernel, **kwargs)
    #solver = overthrust_setup(filename=filename, tn=50, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)
    
    u = TimeFunction(name='u', grid=solver.model.grid, time_order=2, space_order=solver.space_order)
    rec = Receiver(name='rec', grid=solver.model.grid,
                              time_range=solver.receiver.time_range,
                              coordinates=solver.receiver.coordinates.data)
    cp = DevitoCheckpoint([u])
    n_checkpoints = None
    m = solver.model.m
    dt = solver.dt
    v = TimeFunction(name='v', grid=solver.model.grid, time_order=2, space_order=solver.space_order)
    grad = Function(name='grad', grid=solver.model.grid)
    wrap_fw = CheckpointOperator(solver.op_fwd(save=False), src=solver.source, u=u, m=m, rec=rec, dt=dt)
    wrap_rev = CheckpointOperator(solver.op_grad(save=False), u=u, v=v, m=m, rec=rec, dt=dt, grad=grad)

    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2)
    with Timer(factor=1000) as tf:
        wrp.apply_forward()
    with Timer(factor=1000) as tr:
        wrp.apply_reverse()
    print("Forward: %d ms, Reverse: %d ms" % (tf.elapsed, tr.elapsed))


if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument("-so", "--space_order", default=6,
                        type=int, help="Space order of the simulation")
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

    verify(nbpml=args.nbpml, 
        space_order=args.space_order, kernel=args.kernel,
        dse=args.dse, dle=args.dle, filename='overthrust_3D_initial_model.h5')
    
    run(nbpml=args.nbpml, 
        space_order=args.space_order, kernel=args.kernel,
        dse=args.dse, dle=args.dle, filename='overthrust_3D_initial_model.h5')

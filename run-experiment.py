from argparse import ArgumentParser
from examples.seismic.acoustic.acoustic_example import acoustic_setup, smooth10
from devito.logger import info, debug
from examples.checkpointing.checkpoint import DevitoCheckpoint, CheckpointOperator
from examples.seismic import PointSource, Receiver, TimeAxis, RickerSource
from examples.seismic.acoustic import AcousticWaveSolver
from examples.seismic.model import Model
from devito import TimeFunction, Function
import numpy as np
from pyrevolve import Revolver
from pyrevolve.compression import CHUNK_SIZE, SHUFFLE, CNAME
import h5py

class CompressionCheckpoint(DevitoCheckpoint):
    def save(self, ptr, compressor):
        """Overwrite live-data in this Checkpoint object with data found at
        the ptr location."""
        i_ptr_lo = 0
        i_ptr_hi = 0
        for o in self.objects:
            print(np.linalg.norm(o.data))
            i_ptr_hi = i_ptr_hi + o.size
            ptr[i_ptr_lo:i_ptr_hi] = compressor(o.data)
            i_ptr_lo = i_ptr_hi

    def load(self, ptr, decompressor):
        """Copy live-data from this Checkpoint object into the memory given by
        the ptr."""
        i_ptr_lo = 0
        i_ptr_hi = 0
        for o in self.objects:
            i_ptr_hi = i_ptr_hi + o.size
            o.data[:] = decompressor(ptr[i_ptr_lo:i_ptr_hi].reshape(o.shape))
            i_ptr_lo = i_ptr_hi

def from_hdf5(filename, **kwargs):
    debug("Reading file")
    f = h5py.File(filename, 'r')
    origin = kwargs.pop('origin', None)
    if origin is None:
        origin_key = kwargs.pop('origin_key', 'o')
        origin = f[origin_key]

    spacing = kwargs.pop('spacing', None)
    if spacing is None:
        spacing_key = kwargs.pop('spacing_key', 'd')
        spacing = f[spacing_key]
    nbpml = kwargs.pop('nbpml', 20)
    datakey = kwargs.pop('datakey', None)
    if datakey is None:
        raise ValueError("datakey must be known - what is the name of the data in the file?")
    shape = f[datakey].shape
    space_order=kwargs.pop('space_order', None)
    dtype = kwargs.pop('dtype', None)
    data_m = f[datakey][()]
    data_vp = np.sqrt(1/data_m)
    debug("File read complete")
    return Model(space_order=space_order, vp=data_vp, origin=origin, shape=shape,
                     dtype=dtype, spacing=spacing, nbpml=nbpml)

def overthrust_setup(filename, kernel='OT2', space_order=2, nbpml=40, **kwargs):
    model = from_hdf5(filename, space_order=space_order, nbpml=nbpml, datakey='m0', dtype=np.float64)
    spacing = model.spacing
    shape = model.vp.shape
    nrec = shape[0]
    tn = round(2*max(model.domain_size)/np.min(model.vp))
    # Derive timestepping from model spacing
    dt = model.critical_dt * (1.73 if kernel == 'OT4' else 1.0)
    t0 = 0.0
    time_range = TimeAxis(start=t0, stop=tn, step=dt)

    # Define source geometry (center of domain, just below surface)
    src = RickerSource(name='src', grid=model.grid, f0=0.01, time_range=time_range)
    src.coordinates.data[0, :] = np.array(model.domain_size) * .5
    if len(shape) > 1:
        src.coordinates.data[0, -1] = model.origin[-1] + 2 * spacing[-1]
    # Define receiver geometry (spread across x, just below surface)
    rec = Receiver(name='rec', grid=model.grid, time_range=time_range, npoint=nrec)
    rec.coordinates.data[:, 0] = np.linspace(0., model.domain_size[0], num=nrec)
    if len(shape) > 1:
        rec.coordinates.data[:, 1:] = src.coordinates.data[0, 1:]

    # Create solver object to provide relevant operators
    solver = AcousticWaveSolver(model, source=src, receiver=rec, kernel=kernel,
                                space_order=space_order, **kwargs)
    return solver

def run(space_order=4, kernel='OT2', nbpml=40, 
        autotune=False, filename='', chunk=1000000, algo=None, 
        shuffle="SHUFFLE", **kwargs):

    solver = overthrust_setup(filename=filename, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)
    m = solver.model.m
    dt = solver.dt
    u = TimeFunction(name='u', grid=solver.model.grid,
                     time_order=2, space_order=solver.space_order)
    v = TimeFunction(name='v', grid=solver.model.grid,
                     time_order=2, space_order=solver.space_order)
    rec = Receiver(name='rec', grid=solver.model.grid, time_range=solver.receiver.time_range,
                   coordinates=solver.receiver.coordinates.data)
    grad = Function(name='grad', grid=solver.model.grid)
    cp = CompressionCheckpoint([u])
    n_checkpoints = 60
    wrap_fw = CheckpointOperator(solver.op_fwd(save=False), src=solver.source, u=u, m=m,
                                 dt=solver.dt, rec=rec)
    wrap_rev = CheckpointOperator(solver.op_grad(save=False), u=u, v=v,
                                  m=m, rec=rec, dt=dt, grad=grad)

    # Run forward
    wrp = Revolver(cp, wrap_fw, wrap_rev, n_checkpoints, rec.data.shape[0]-2, compression='blosc', compression_params={CHUNK_SIZE: chunk, CNAME: algo, SHUFFLE: shuffle})
    info("Applying Forward")
    solver.forward(time=100)
    #raw_fw(dt=dt)
    info("Again")
    wrp.apply_forward()
    print(np.linalg.norm(u.data))
    print(np.linalg.norm(rec.data))
    info("Applying Gradient")
    summary = wrp.apply_reverse()
    print("Gradient is: %d" % np.linalg.norm(grad.data))


if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument('-c', dest='chunk', default=1000000, type=int,
                        help="Compression chunk size")
    parser.add_argument('-s', dest='shuffle', default="SHUFFLE", type=str,
                        help="Compression shuffling")
    parser.add_argument('-z', dest='algo', default="blosclz", type=str,
                        help="Compression chunk size")
    parser.add_argument('-a', '--autotune', default=False, action='store_true',
                        help="Enable autotuning for block sizes")
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
    parser.add_argument("--constant", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    parser.add_argument("--checkpointing", default=False, action='store_true',
                        help="Constant velocity model, default is a two layer model")
    args = parser.parse_args()


    run(nbpml=args.nbpml, 
        space_order=args.space_order, kernel=args.kernel,
        autotune=args.autotune, dse=args.dse, dle=args.dle, filename='overthrust_3D_initial_model.h5', chunk=args.chunk, shuffle=args.shuffle, algo=args.algo)

from argparse import ArgumentParser
from simple import overthrust_setup
import h5py
from devito import TimeFunction
import numpy as np


def run(original, lossy, filename='', space_order=4, kernel='OT2', nbpml=40, **kwargs):

    solver = overthrust_setup(filename=filename, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)

    with h5py.File(original) as original_file:
        correct_u = original_file['data'][()]
    with h5py.File(lossy) as lossy_file:
        lossy_u = lossy_file['data'][()]
    u = TimeFunction(name="u", grid=solver.model.grid, time_order=2, space_order=solver.space_order)
    errors = []
    for steps in range(1, 1000, 10):
        u.data[:] = correct_u[:]
        rec, u_correct_p, summary = solver.forward(save=False, u=u, time=steps)
        u_correct_p = u_correct_p.data.copy()
        u.data[:] = lossy_u[:]
        rec, u_lossy_p, summary = solver.forward(save=False, u=u, time=steps)
        u_lossy_p = u_lossy_p.data.copy()
        error = u_correct_p - u_lossy_p
        it_errors = (steps, np.linalg.norm(error), np.max(error.data))
        print(it_errors)
        errors.append(it_errors)
    print(errors)


if __name__ == "__main__":
    description = ("Example script for a set of acoustic operators.")
    parser = ArgumentParser(description=description)
    parser.add_argument("original", type=str)
    parser.add_argument("lossy", type=str)
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

    run(original=args.original, lossy=args.lossy, nbpml=args.nbpml, 
        space_order=args.space_order, kernel=args.kernel,
        dse=args.dse, dle=args.dle, filename='overthrust_3D_initial_model.h5')

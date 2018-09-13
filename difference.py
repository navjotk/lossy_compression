from argparse import ArgumentParser
from simple import overthrust_setup
import h5py

def run(original, lossy, space_order=4, kernel='OT2', nbpml=40, **kwargs):

    solver = overthrust_setup(filename=filename, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)

    with h5py.File(original) as original_file:
        correct_u = original_file['data'][()]
    with h5py.File(lossy) as lossy_file:
        lossy_u = lossy_file['data'][()]

    errors = []
    for steps in range(1, 1000, 10):
        rec, u_correct_p, summary = solver.forward(save=False, u=correct_u, time=steps)
        rec, u_lossy_p, summary = solver.forward(save=False, u=lossy_u, time=steps)
        error = u_correct_p.data - u_lossy_p.data
        errors.append((steps, np.linalg.norm(error), np.max(error)))
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
        dse=args.dse, dle=args.dle)

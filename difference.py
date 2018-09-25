from argparse import ArgumentParser
from simple import overthrust_setup
import h5py
from devito import TimeFunction
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_field(u):
    return plt.imshow(np.transpose(u[0]),
                    animated=True, vmin=-1e0, vmax=1e0,
                    cmap=cm.RdGy)

def init_video(solver, correct_u, lossy_u):
    model = solver.model
    fig = plt.figure(figsize=(24, 10))
    im_correct_u = plot_field(correct_u)
    plt.xlabel('X position (km)',  fontsize=20)
    plt.ylabel('Depth (km)',  fontsize=20)
    plt.tick_params(labelsize=20)
    im_model = plt.imshow(np.transpose(model.vp[0]), vmin=1.5, vmax=4.5, cmap=cm.jet,
                 alpha=.4)
    im_lossy_u = plot_field(lossy_u)
    return fig, im_correct_u, im_lossy_u, im_model

def finalize_video(fig, ims, filename):
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename)

def run(original, lossy, filename='', space_order=4, kernel='OT2', nbpml=40, **kwargs):

    solver = overthrust_setup(filename=filename, nbpml=nbpml, space_order=space_order, kernel=kernel, **kwargs)
    # Load two timesteps of each
    # Uncompressed should be saved by simple.py
    # Compressed one should be generated on demand by makefile
    # (dedicated script to convert uncompressed previous timestep to compressed)
    with h5py.File(original) as original_file:
        correct_u = original_file['data'][()]
    with h5py.File(lossy) as lossy_file:
        lossy_u = lossy_file['data'][()]

    u = TimeFunction(name="u", grid=solver.model.grid, time_order=2, space_order=solver.space_order)
    fig, im_correct_u, im_lossy_u, im_model = init_video(solver, correct_u, lossy_u)
    im_corrects = [[im_correct_u, im_model]]
    im_lossys = [[im_lossy_u, im_model]]
    errors = []
    print("steps, error_norm, max_error")
    for steps in range(1, 50, 10):
        u.data[:] = correct_u[:]
        ini_t = 2000
        rec, u_correct_p, summary = solver.forward(save=False, u=u, time_M=(ini_t+steps), time_m=ini_t)
        u_correct_p = u_correct_p.data[(ini_t+steps)%3].copy()
        u.data[:] = lossy_u[:]
        rec, u_lossy_p, summary = solver.forward(save=False, u=u, time_M=(ini_t+steps), time_m=ini_t)
        u_lossy_p = u_lossy_p.data[(ini_t+steps)%3].copy()
        error = u_correct_p - u_lossy_p
        it_errors = (steps, np.linalg.norm(error), np.max(error.data))
        print(", ".join([str(x) for x in it_errors]))
        errors.append(it_errors)
        im_corrects.append([plot_field(u_correct_p), im_model])
        im_lossys.append([plot_field(u_lossy_p), im_model])

    finalize_video(fig, im_corrects, original+".mp4")
    finalize_video(fig, im_lossys, lossy+".mp4")


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

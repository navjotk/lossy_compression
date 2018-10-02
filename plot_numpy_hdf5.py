import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import h5py
import numpy as np
from simple import from_hdf5

input_file = sys.argv[1]
output_file = sys.argv[2]

with h5py.File(input_file) as ifile:
    data = ifile['data'][()]

velocity_model='overthrust_3D_initial_model.h5'

model = from_hdf5(velocity_model, space_order=2, nbpml=20, datakey='m0', dtype=np.float32)

vmax = np.max(data)

slice_loc = 440
plt.imshow(np.transpose(data[slice_loc]), vmin=-vmax, vmax=vmax, cmap="seismic")
plt.imshow(np.transpose(model.vp[slice_loc]), alpha=0.5)
plt.colorbar()
plt.savefig(output_file)

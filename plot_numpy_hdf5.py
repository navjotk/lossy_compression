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
shape = model.vp.shape
vmax = np.max(data)

if len(data.shape)==3:
    slice_loc = 440
    im = plt.imshow(np.transpose(data[slice_loc]), vmax=.01, vmin=-.01, cmap="seismic",
           extent = [0, 20, 0.001*(shape[-1]-1)*25, 0])
else:
    im = plt.imshow(np.transpose(data), vmax=vmax, vmin=-vmax, cmap="seismic")
plt.xlabel("X (km)")
plt.ylabel("Depth (km)")
cb = plt.colorbar(shrink=.3, pad=.01, aspect=10)
for l in cb.ax.yaxis.get_ticklabels():
    l.set_fontsize(12)

cb.set_label('Pressure')

plt.savefig(output_file, bbox_inches='tight')

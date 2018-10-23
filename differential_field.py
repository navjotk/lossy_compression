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
    lossy_field = ifile['data'][()]

original_field_file='uncompressed.h5'

with h5py.File(original_field_file) as ofile:
    original_field = ofile['data'][()]

data = lossy_field - original_field
shape = data.shape
vmax = max(np.max(data), -np.min(data))

slice_loc = 440
im = plt.imshow(np.transpose(data[slice_loc]), vmax=vmax, vmin=-vmax, cmap="seismic",
           extent = [0, 20, 0.001*(shape[-1]-1)*25, 0])

plt.xlabel("X (km)")
plt.ylabel("Depth (km)")
cb = plt.colorbar(shrink=.4, pad=.01, aspect=10)
for l in cb.ax.yaxis.get_ticklabels():
    l.set_fontsize(12)

cb.set_label('Absolute error')

plt.savefig(output_file, bbox_inches='tight')

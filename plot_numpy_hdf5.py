import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import h5py
import numpy as np


input_file = sys.argv[1]
output_file = sys.argv[2]

with h5py.File(input_file) as ifile:
    data = ifile['data'][()]

vmax = np.max(data)

plt.imshow(data[143], vmin=-vmax, vmax=vmax, cmap="seismic")

plt.savefig(output_file)

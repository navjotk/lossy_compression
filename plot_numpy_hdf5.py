import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import h5py


input_file = sys.argv[1]
output_file = sys.argv[2]

with h5py.File(input_file) as ifile:
    data = ifile['data'][()]

plt.imshow(data[143], cmap="seismic")

plt.savefig(output_file)

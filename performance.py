from pyrevolve import Operator, Checkpoint, Revolver

from itertools import product
import math
import numpy as np
import matplotlib.pyplot as plt
import csv


class CounterOperator(Operator):
    def __init__(self):
        self.counter = 0

    def apply(self, *args, **kwargs):
        t_s = kwargs['t_start']
        t_e = kwargs['t_end']
        self.counter += abs(t_s - t_e)

class CounterCheckpoint(Checkpoint):
    def __init__(self):
        self.savecount = 0
        self.loadcount = 0

    def save(self, *args):
        self.savecount += 1

    def load(self, *args):
        self.loadcount += 1

    @property
    def size(self):
        return 1

    @property
    def dtype(self):
        return np.int32
    

def revolve_factors(nt, ncp):
    fwd = CounterOperator()
    rev = CounterOperator()
    cp = CounterCheckpoint()
    revolver = Revolver(cp, fwd, rev, ncp, nt)
    revolver.apply_forward()

    assert(fwd.counter == nt)
    revolver.apply_reverse()
    assert(rev.counter == nt)
    return fwd.counter, cp.savecount, cp.loadcount

def get_compressions(filename):
    values = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader, None)
        for row in reader:
            values.append((float(row[4]), float(row[1]), float(row[2])/1000, float(row[3])/1000))
    return values

class Problem(object):
    def __init__(self, nt, size_ts, compute_ts, bw):
        # Number of timesteps
        self.nt = nt
        # Size (MB) of one timestep
        self.size_ts = size_ts
        # Time taken to compute a single timestep (s)
        self.compute_ts = compute_ts
        # Memory bandwidth (MB/s), usually measured with the stream benchmark
        self.bw = bw

    def naive_strategy(self):
        time = 2*self.nt*self.compute_ts
        peak_memory = self.nt*self.size_ts
        return time, peak_memory

    def compression_only(self, c_time, d_time):
        time_fw = self.nt*self.compute_ts
        time_rev = self.nt*self.compute_ts
        time_compression = self.nt * (c_time + d_time)
        return time_fw, time_rev, time_compression

    @property
    def time_copy_ts(self):
        return self.size_ts/self.bw

    def revolve(self, peak_memory):
        ncp = math.floor(peak_memory/self.size_ts)
        advances, takeshots, useshots = revolve_factors(self.nt, ncp)
        assert(advances >= self.nt)
        time_fw = self.compute_ts * advances
        time_rev = self.compute_ts * self.nt
        time_storage = (takeshots + useshots) * self.time_copy_ts
        return time_fw, time_rev, time_storage

    def revolve_compression(self, peak_memory, c_factor, c_time, d_time):
        compressed_size = self.size_ts/c_factor
        ncp = math.floor(peak_memory/compressed_size)

        if ncp >= self.nt:
            return self.compression_only(c_time, d_time)
        
        # If we're here, we can't fit the whole history in memory even after compression
        advances, takeshots, useshots = revolve_factors(self.nt, ncp)
        time_fw = self.compute_ts * advances
        time_rev = self.compute_ts * self.nt

        # Same assumption about 1 store = 1 read as above
        time_storage = (takeshots * c_time) + (useshots * d_time)
        return time_fw, time_rev, time_storage

    def compression_speedup(self, peak_memory, c_factor, c_time, d_time):
        c_fw, c_r, c_s = self.revolve_compression(peak_memory, c_factor, c_time, d_time)
        total_time_compression = c_fw + c_r + c_s

        r_fw, r_r, r_s = self.revolve(peak_memory)
        total_time_revolve = r_fw + r_r + r_s

        return total_time_revolve/total_time_compression


def linspace(lower, upper, length):
    return [lower + x*(upper-lower)/length for x in range(length)]


def platform_name():
    global platform
    return platform

def plot(x, y, filename, title, xlabel, ylabel, hline=None, more_y=None, more_y_labels=None,
         fixed=None, xscale=None, yscale=None, vlines=None, textloc=None):
    plt.gcf().clear()
    if more_y_labels is not None:
        # We need labels for every series in more_y and one for the main series
        assert(len(more_y_labels) == len(more_y) + 1)
        label = more_y_labels[0]
        more_y_labels = more_y_labels[1:]
    else:
        label = None
    plt.plot(x, y, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
    plt.xticks(rotation=90)
    if hline is not None:
        plt.axhline(hline, linestyle='dashed', color='red')
    if more_y is not None:
        for i, series in enumerate(more_y):
            if more_y_labels is not None:
                label = more_y_labels[i]
            else:
                label = None
            plt.plot(x, series, linestyle='dotted', label=label)
        if more_y_labels is not None:
            plt.legend()
    
    if fixed is not None:
        fixed.update({"Platform" : platform_name()})
        fixed_params = ", \n".join(["%s: %s" % (key, value) for key, value in fixed.items()])
        if textloc is None:
            textloc = [0.5, 0.5]
        plt.figtext(textloc[0], textloc[1], fixed_params, size=8)
    plt.title(title)
    if xscale is not None:
        plt.xscale(xscale)

    if yscale is not None:
        plt.yscale(yscale)
    if vlines is not None:
        for v in vlines:
            plt.axvline(v, linestyle='-.', color='green')
            plt.text(v + 100,2.25,str(v),rotation=90)
    plt.savefig(filename, bbox_inches='tight')


    
# Plots

# Problem params
nt = 2526
cp_size = 287*881*881*4/1000000

#System params
compute_ts = 1.11
#bandwidth = 6777 # Richter
bandwidth = 8139.2 # Skylake
platform = "Skylake" # Richter

# Compressions Params
c_factor = 41
c_time = 0.36
d_time = 1.6
t_d_time = 0.396

# Plot params
peak_mem = 4000
more_peak_mem = [8000, 16000, 24000]

compression_filename="tolerance-richter.csv"
compression_label = "Tolerance"

# (Speedup vs) varying peak memory

def varying_peak_memory(nt, size_ts, compute_ts, bw, c_factor, c_time, d_time,
                        theoretical_d_time):
    p = Problem(nt, size_ts, compute_ts, bw)
    min_time, overall_peak_mem = p.naive_strategy()
    mems = linspace(4*size_ts, overall_peak_mem, 200)
    speedups = [p.compression_speedup(x, c_factor, c_time, d_time) for x in mems]
    theoretical_speedup = [p.compression_speedup(x, c_factor, c_time, theoretical_d_time) for x in mems]
    fixed = {'Timesteps': nt, 'Size of checkpoint (MB)': size_ts,
             'Time for compute step (s)': compute_ts, 'Bandwidth (MB/s)': bw,
             'Compression Factor': c_factor, 'Compression Time (s)': c_time,
             'Decompression time (s)': d_time,
             'Theoretical decompression time (s)': theoretical_d_time}
    plot(mems, speedups, "varying-memory.pdf", "Speedup for varying peak memory", "Memory (MB)",
             "Speedup (x)", hline=1, vlines=[20000, 52000], more_y=[theoretical_speedup], fixed=fixed, xscale='log', textloc=[0.5, 0.5])

varying_peak_memory(nt, cp_size, compute_ts, #time to compute one timestep (s)
                    bandwidth, # Memory bandwidth from stream (MB/s)
                    c_factor, # Compression factor
                    c_time, # Compression time(s)
                    d_time, # Decompression time(s)
                    t_d_time, # Theoretical decompression time (s)
                    )

# (Speedup vs) varying AI. Fixed compression.
def varying_compute(nt, size_ts, peak_mem, bw, c_factor, c_time, d_time, more_peak_mem=None):
    computes = linspace(0, 100, 200)
    speedups = []
    
    if more_peak_mem is not None:
        more_speedups = [[] for x in more_peak_mem]
    else:
        more_speedups = None
    for compute_ts in computes:
        p = Problem(nt, size_ts, compute_ts, bw)
        speedups.append(p.compression_speedup(peak_mem, c_factor, c_time, d_time))
        if more_peak_mem is not None:
            for i, pm in enumerate(more_peak_mem):
                more_speedups[i].append(p.compression_speedup(pm, c_factor, c_time, d_time))
    fixed = {'Timesteps': nt, 'Size of checkpoint (MB)': size_ts,
             'Peak memory (MB)': peak_mem, 'Bandwidth (MB/s)': bw,
             'Compression Factor': c_factor, 'Compression Time (s)': c_time,
             'Decompression time (s)': d_time,}
    plot(computes, speedups, "varying-compute.pdf",
         "Speedup for varying compute/timestep", "Compute time (s)", "Speedup (x)",
         hline=1, more_y=more_speedups,
         more_y_labels=["Peak memory: %d" % x for x in [peak_mem] + more_peak_mem],
         fixed=fixed, xscale='log', textloc=[0.55, 0.55])

varying_compute(nt, cp_size, peak_mem, # Peak memory
                                   bandwidth, # Memory bandwidth from stream (MB/s)
                    c_factor, # Compression factor
                    c_time, # Compression time(s)
                    d_time, # Decompression time(s)
                    more_peak_mem=more_peak_mem
                    )

# (Speedup vs) compression ratios (and times)

def varying_compression(nt, size_ts, compute_ts, peak_mem, bw, filename, label):
    compressions = get_compressions(filename) # list of (x, c_factor, c_time, d_time)
    x_param = [x[0] for x in compressions]
    p = Problem(nt, size_ts, compute_ts, bw)

    speedups = [p.compression_speedup(peak_mem, f, c, d) for _, f, c, d in compressions]
    fixed = {'Timesteps': nt, 'Size of checkpoint (MB)': size_ts,
             'Time for compute step (s)': compute_ts, 'Bandwidth (MB/s)': bw,
             'Peak memory (MB)': peak_mem}
    plot(x_param, speedups, "varying-compression.png",
         "Speedup for varying compression parameters", label, "Speedup (x)", hline=1,
         fixed=fixed, xscale='log')
    
#varying_compression(nt, cp_size, compute_ts, #time to compute one timestep (s)
#                    peak_mem, # Peak memory
#                    bandwidth, # Memory bandwidth from stream (MB/s)
#                    compression_filename, # File with compression data
#                    compression_label
#                    )
    
# # For overthrust devito
# # For NN
# # For something with really high AI


def varying_nt(size_ts, compute_ts, peak_mem, bw, f, c, d):
    nts = linspace(2000, 20000, 200)
    nts = [math.floor(x) for x in nts]
    speedups = []
    for nt in nts:
        p = Problem(nt, size_ts, compute_ts, bw)
        speedups.append(p.compression_speedup(peak_mem, f, c, d))

    fixed = {'Size of checkpoint (MB)': size_ts,
             'Time for compute step (s)': compute_ts, 'Bandwidth (MB/s)': bw,
             'Peak memory (MB)': peak_mem,
             'Compression Factor': c_factor, 'Compression Time (s)': c_time,
             'Decompression time (s)': d_time,}
    plot(nts, speedups, "varying-nt.pdf", "Speedup for varying number of timesteps", "Timesteps","Speedup (x)", hline=1, fixed=fixed, xscale='log', textloc=[0.5, 0.4])
    
varying_nt(cp_size, compute_ts, #time to compute one timestep (s)
                    peak_mem, # Peak memory
                    bandwidth, # Memory bandwidth from stream (MB/s)
                    c_factor, c_time, d_time
                    )
#Where is the tipping point?

# Gives a speedup for everything that is costly and for cheap stuff, when memory is very constrained


def vary_all(size_ts, bw, f, c, d):
    nts = linspace(2000, 20000, 200)
    nts = [math.floor(x) for x in nts]
    computes = linspace(0, 100, 200)

    results = np.zeros((200, 200, 200))

    for i, nt in enumerate(nts):
        for j, compute in enumerate(computes):
            p = Problem(nt, size_ts, compute, bw)
            min_time, overall_peak_mem = p.naive_strategy()
            mems = linspace(2*size_ts, overall_peak_mem/c_factor, 200)
            for k, mem in enumerate(mems):
                results[i, j, k] = p.compression_speedup(mem, f, c, d)

    np.save('results', results)

#vary_all(cp_size, bandwidth, c_factor, c_time, d_time)
    

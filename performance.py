from pyrevolve import Operator, Checkpoint, Revolver

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

class AIProblem(object):
    def __init__(self, nt, size_ts, ai, compute_speed, bw):
        compute_ts = (ai * size_ts) / compute_speed
        super(AIProblem, self).__init__(nt, size_ts, compute_ts, bw)


def linspace(lower, upper, length):
    return [lower + x*(upper-lower)/length for x in range(length)]

def plot(x, y, filename, title, xlabel, ylabel, hline=None, more_y=None, more_y_labels=None):
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
    plt.title(title)
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
    plt.savefig(filename, bbox_inches='tight')

# Plots

# (Speedup vs) varying peak memory

def varying_peak_memory(nt, size_ts, compute_ts, bw, c_factor, c_time, d_time,
                        theoretical_d_time):
    p = Problem(nt, size_ts, compute_ts, bw)
    min_time, overall_peak_mem = p.naive_strategy()
    mems = linspace(2*size_ts, overall_peak_mem/c_factor, 200)
    speedups = [p.compression_speedup(x, c_factor, c_time, d_time) for x in mems]
    theoretical_speedup = [p.compression_speedup(x, c_factor, c_time, theoretical_d_time) for x in mems]
    plot(mems, speedups, "varying-memory.png", "Speedup for varying peak memory", "Memory (MB)",
             "Speedup (x)", hline=1, more_y=[theoretical_speedup])

#varying_peak_memory(2000, 287*881*881*4/1000000, 2.276333333, #time to compute one timestep (s)
#                    7234, # Memory bandwidth from stream (MB/s)
#                    29.72, # Compression factor
#                    1.45, # Compression time(s)
#                    1.65, # Decompression time(s)
#                    0.7 # Theoretical decompression time (s)
#                    )

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
    plot(computes, speedups, "varying-compute.png", "Speedup for varying compute time/timestep", "Compute time (s)",
             "Speedup (x)", hline=1, more_y=more_speedups, more_y_labels=["Peak memory: %d" % x for x in [peak_mem] + more_peak_mem])

#varying_compute(2000, 287*881*881*4/1000000, 8000, # Peak memory
#                                   7234, # Memory bandwidth from stream (MB/s)
#                    29.72, # Compression factor
#                    1.45, # Compression time(s)
#                    1.65, # Decompression time(s)
#                    more_peak_mem=[16000, 24000, 32000]
#                    )

# (Speedup vs) compression ratios (and times)

def varying_compression(nt, size_ts, compute_ts, peak_mem, bw, filename, label):
    compressions = get_compressions(filename) # list of (x, c_factor, c_time, d_time)
    x_param = [x[0] for x in compressions]
    p = Problem(nt, size_ts, compute_ts, bw)

    speedups = [p.compression_speedup(peak_mem, f, c, d) for _, f, c, d in compressions]
    plot(x_param, speedups, "varying-compression.png", "Speedup for varying compression parameters", label,
             "Speedup (x)", hline=1)
    
varying_compression(2000, 287*881*881*4/1000000, 2.276333333, #time to compute one timestep (s)
                    8000, # Peak memory
                    7234, # Memory bandwidth from stream (MB/s)
                    "tolerance-parallel.csv", # File with compression data
                    "Tolerance"
                    )
    
# # For overthrust devito
# # For NN
# # For something with really high AI

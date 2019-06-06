from performance import Problem
from plot_timings import consolidated_results_from_file
import sys
import matplotlib.pyplot as plt


def revolve_theoretical(nt, size_ts, compute_ts, bw):
    p = Problem(nt, size_ts, compute_ts, bw)
    min_time, overall_peak_mem = p.naive_strategy()
    mems = [x*size_ts for x in list(range(2, 50))]
    times = [sum(p.revolve(x)) for x in mems]
    return times, mems

def revolve_actual(filename):
    results, consolidated_results, field_names, ncp = consolidated_results_from_file(filename)
    total_exec_times = [sum(results[n].values()) for n in ncp]
    return total_exec_times, ncp

nt = 631
size_ts = 2*287*881*881*4/1000000
#No compression
compute_ts = 550
bw = 1.9

#Compression OT2
compute_ts = 550
bw = 1.2

#Compression OT4
compute_ts = 1000
bw = 1.2

filename = "model_validation_compression_ot4.pdf"

measured_exec_times, ncp = revolve_actual(sys.argv[1])
measured_mems = [x*size_ts for x in ncp]
predicted_times, mems = revolve_theoretical(nt, size_ts, compute_ts, bw)

plt.plot(mems, predicted_times, label="Predicted")
plt.plot(measured_mems, measured_exec_times, label="Measured")
plt.yscale('log')
plt.xscale('log', basex=2)
plt.legend()
plt.xlabel("Peak memory (MB)")
plt.ylabel("Total time to solution (ms)")
plt.title("Comparison of predicted and measured runtimes for OT4 kernel with compression")
plt.savefig(filename, bbox_inches='tight')

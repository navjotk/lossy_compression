from performance import Problem
from plot_timings import consolidated_results_from_file
import sys


def revolve_theoretical(nt, size_ts, compute_ts, bw):
    p = Problem(nt, size_ts, compute_ts, bw)
    min_time, overall_peak_mem = p.naive_strategy()
    mems = range(2, 50) * size_ts
    times = [sum(p.revolve(x)) for x in mems]
    return times, mems

def revolve_actual(filename):
    results, consolidated_results, field_names, ncp = consolidated_results_from_file(filename)
    total_exec_times = [sum(results[n].values()) for n in ncp]
    return total_exec_times


revolve_actual(sys.argv[1])

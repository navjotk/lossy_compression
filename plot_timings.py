import csv
import sys
import seaborn as sns
import matplotlib.pyplot as plt

width = 0.8


results_file = sys.argv[1]
INF = 9999999999
results = {}

def consolidated_results_from_file(filename):
    with open(results_file,'r') as fd:
        reader = csv.DictReader(fd)
        for row in reader:
            ncp = int(row['ncp'])
            ncp_dict = results.get(ncp, {})
            for k, v in row.items():
                if k in ("forward_takeshot_timing", "forward_advance_timing", "forward_lastfw_timing", "reverse_reverse_timing", "reverse_restore_timing", "reverse_advance_timing",
                     "reverse_takeshot_timing"):
                    v = float(v)
                    existing = ncp_dict.get(k, INF)
                    if v < existing:
                        ncp_dict[k] = v
                        results[ncp] = ncp_dict

    legends = []

    ncp = list(sorted(results.keys()))

    field_names = list(results[ncp[0]].keys())

    consolidated_results = [[results[cp][x] for cp in ncp] for x in field_names ]

    return results, consolidated_results, field_names, ncp

if __name__ == "__main__":
    consolidated_results, field_names = consolidated_results_from_file(results_file)

    events_blue = ("forward_takeshot_timing", "reverse_restore_timing", "reverse_takeshot_timing")
    events_green = ("forward_advance_timing", "forward_lastfw_timing", "reverse_reverse_timing", "reverse_advance_timing")

    bluep = ["lightsteelblue", "cornflowerblue", "royalblue", "slateblue", "blueviolet"]
    greenp = ["forestgreen", "orange", "limegreen", "lightcoral", "chocolate"]
    bluecounter = 0
    greencounter = 0
    plots = []
    for i, name in enumerate(field_names):
        if name in events_blue:
            cm = bluep[bluecounter]
            bluecounter+=1
        elif name in events_green:
            cm = greenp[greencounter]
            greencounter+=1
        else:
            cm = None
        if i<1:
            p = plt.bar(ncp, consolidated_results[i], width, color=cm)
        else:
            b = [consolidated_results[k] for k in range(i)]
            b = [sum(i) for i in zip(*b)]
            p = plt.bar(ncp, consolidated_results[i], width, bottom=b, color=cm)
        plots.append(p[0])

    filename = "revolve_timings.png"
    plt.legend(plots, field_names, loc='lower center', bbox_to_anchor=(0.5, 1.05),
          ncol=3, fancybox=True, shadow=True)
    plt.savefig(filename, bbox_inches='tight')

#for ncp, values in results.items():
#    for r_name, r_val in values.items():
#        p = plt.bar(ncp, r_val, width)
#        legends.append((p[0], r_name))

#plt.legend(legends)
#plt.xlabel('Number of checkpoints (n)')
#plt.ylabel('Time to solution (ms)')
#plt.show()

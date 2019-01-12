import sys
from ast import literal_eval
from plotter import plot

infile = sys.argv[1]

data = []
with open(infile) as datafile:
    for i, row in enumerate(datafile):
        data.append(literal_eval(row))

indices, ratios, times = zip(*data)


plot(indices, ratios, filename='compression-ratios-through-simulation.pdf', xlabel='Simulation time (n)', ylabel='Compression ratio (x)', title='Variation of achievable compression ratio as the simulation progresses', xscale='linear')
        

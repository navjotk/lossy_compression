import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import csv


def read_csv_file(filename):
    with open(filename, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',', quotechar='"')
        headings = []
        data = []
        for i, row in enumerate(csv_reader):
            if i == 0:
                headings = [a.replace('"', '') for a in row]
            else:
                data.append([float(x) for x in row])
    data = list(zip(*data))
    return data, headings
        

def plot(x, y, filename, title, xlabel, ylabel, xscale='log', yscale='log', highlight=None):
    plt.plot(x, y, marker='.')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.axhline(1, linestyle='dashed', color='red')
    if highlight is not None:
        for i in highlight:
            plt.plot(x[i], y[i], marker='^')
    plt.savefig(filename, bbox_inches='tight')

if __name__ == "__main__":
    description = ("Plotter")
    parser = ArgumentParser(description=description)
    parser.add_argument("filename", type=str)
    parser.add_argument("xindex", type=int)
    parser.add_argument("yindex", type=int)
    parser.add_argument("title", type=str)
    parser.add_argument("output_file", type=str)
    args = parser.parse_args()

    filename = args.filename
    xindex = args.xindex
    yindex = args.yindex
    title = args.title
    output_file = args.output_file
    data, headings = read_csv_file(filename)
    plot(data[xindex], data[yindex], output_file, title, headings[xindex], headings[yindex], highlight=[3])

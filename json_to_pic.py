import json
import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt


our_times = []
no_allo_times = []
no_lr_times = []


cwd = Path.cwd()

def sort(x):
    return sorted(x, key=lambda x: x[0])

fns = cwd.glob("all-*.json")
for fn in fns:
    threshold = int(fn.name[4:-5])
    with open(fn) as f:
        j = json.load(f)
        our_times.append((threshold, float(j["real time"])))

fns = cwd.glob("no-allo-*.json")
for fn in fns:
    threshold = int(fn.name[8:-5])
    with open(fn) as f:
        j = json.load(f)
        no_allo_times.append((threshold, float(j["real time"])))

fns = cwd.glob("no-lr-*.json")
for fn in fns:
    threshold = int(fn.name[6:-5])
    with open(fn) as f:
        j = json.load(f)
        no_lr_times.append((threshold, float(j["real time"])))

fig, ax = plt.subplots()

our_times = sort(our_times)
no_allo_times = sort(no_allo_times)
no_lr_times = sort(no_lr_times)

ax.plot(*list(zip(*our_times)))
ax.plot(*list(zip(*no_allo_times)))
ax.plot(*list(zip(*no_lr_times)))

plt.savefig('foo.png')


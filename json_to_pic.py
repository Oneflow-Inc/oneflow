import json
import csv
import sys
import re
from pathlib import Path
import matplotlib.pyplot as plt


our_times = []
nlr_times = []
no_allo_times = []
no_lr_times = []


cwd = Path.cwd()


def draw_from_file(fn_pattern, threshold_fn, time_fn, label):
    fns = list(filter(re.compile(fn_pattern).match, (str(x.name) for x in cwd.iterdir())))
    data = []
    for fn in fns:
        threshold = threshold_fn(fn)
        time = time_fn(fn)
        data.append((threshold, time))
    data = sorted(data, key=lambda x: x[0])
    ax.plot(*list(zip(*data)), label=label)


def get_theo_time_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        return float(j["theoretically time"])


def get_real_time_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        return float(j["real time"])


def get_threshold_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        return float(j["threshold"][:-2])


fig, ax = plt.subplots()

draw_from_file(r"resnet152-normal-\d+.json", get_threshold_from_json_file, get_theo_time_from_json_file, 'our')
draw_from_file(r"resnet152-nlr-\d+.json", get_threshold_from_json_file, get_theo_time_from_json_file, 'nlr')
draw_from_file(r"resnet152-no-allo-\d+.json", get_threshold_from_json_file, get_theo_time_from_json_file, 'no-allo')
draw_from_file(r"resnet152-no-lr-\d+.json", get_threshold_from_json_file, get_theo_time_from_json_file, 'no-lr')
# draw_from_file(r"no-lr-no-fbip-\d+.json", lambda fn: int(fn[14:-5]), get_theo_time_from_json_file, 'no-lr-no-fbip')
# draw_from_file(r"nlr-no-fbip-\d+.json", lambda fn: int(fn[12:-5]), get_theo_time_from_json_file, 'nlr-no-fbip')

ax.legend()

plt.savefig('resnet152-theo-time.png')

fig, ax = plt.subplots()

draw_from_file(r"resnet152-normal-\d+.json", get_threshold_from_json_file, get_real_time_from_json_file, 'our')
draw_from_file(r"resnet152-nlr-\d+.json", get_threshold_from_json_file, get_real_time_from_json_file, 'nlr')
draw_from_file(r"resnet152-no-allo-\d+.json", get_threshold_from_json_file, get_real_time_from_json_file, 'no-allo')
draw_from_file(r"resnet152-no-lr-\d+.json", get_threshold_from_json_file, get_real_time_from_json_file, 'no-lr')
# draw_from_file(r"no-lr-no-fbip-\d+.json", lambda fn: int(fn[14:-5]), get_real_time_from_json_file, 'no-lr-no-fbip')
# draw_from_file(r"nlr-no-fbip-\d+.json", lambda fn: int(fn[12:-5]), get_real_time_from_json_file, 'nlr-no-fbip')

ax.legend()

plt.savefig('resnet152-real-time.png')


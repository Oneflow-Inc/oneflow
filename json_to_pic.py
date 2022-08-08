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


def draw_from_file(ax, fn_pattern, threshold_fn, time_fn, label):
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


def get_mem_frag_rate_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        mem_frag_rate = j["mem frag rate"]
        if mem_frag_rate is not None:
            mem_frag_rate = float(mem_frag_rate)
        return mem_frag_rate


def get_dataset_time_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        mem_frag_rate = j["dataset time"]
        if mem_frag_rate is not None:
            mem_frag_rate = float(mem_frag_rate)
        return mem_frag_rate


def get_threshold_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        return float(j["threshold"][:-2])


fig, ax = plt.subplots()

model_name = sys.argv[1]

def draw_from_files_and_draw(get_y, pic_name):
    _, ax = plt.subplots()

    draw_from_file(ax, rf"{model_name}-ours-\d+.json", get_threshold_from_json_file, get_y, 'g2')
    # draw_from_file(ax, rf"{model_name}-ours-g4-\d+.json", get_threshold_from_json_file, get_y, 'g4')
    # draw_from_file(ax, rf"{model_name}-ours-g6-\d+.json", get_threshold_from_json_file, get_y, 'g6')
    draw_from_file(ax, rf"{model_name}-ours-id-guided-\d+.json", get_threshold_from_json_file, get_y, 'g2i')
    # draw_from_file(ax, rf"{model_name}-ours-g4-id-guided-\d+.json", get_threshold_from_json_file, get_y, 'g4i')
    # draw_from_file(ax, rf"{model_name}-ours-g6-id-guided-\d+.json", get_threshold_from_json_file, get_y, 'g6i')
    # draw_from_file(ax, rf"{model_name}-ours-with-size-\d+.json", get_threshold_from_json_file, get_y, 'ours w/ size')
    # draw_from_file(ax, rf"{model_name}-raw-gp-\d+.json", get_threshold_from_json_file, get_y, 'raw grouping')
    # draw_from_file(ax, rf"{model_name}-raw-gp-size-\d+.json", get_threshold_from_json_file, get_y, 'raw grouping w/ size')
    draw_from_file(ax, rf"{model_name}-no-gp-\d+.json", get_threshold_from_json_file, get_y, 'no grouping')
    # draw_from_file(ax, rf"{model_name}-no-gp-size-\d+.json", get_threshold_from_json_file, get_y, 'no grouping w/ size')
    draw_from_file(ax, rf"{model_name}-no-fbip-\d+.json", get_threshold_from_json_file, get_y, 'no fbip')
    # draw_from_file(ax, rf"{model_name}-no-fbip-size-\d+.json", get_threshold_from_json_file, get_y, 'no fbip w/ size')
    draw_from_file(ax, rf"{model_name}-me-style-\d+.json", get_threshold_from_json_file, get_y, 'me style')
    # draw_from_file(ax, rf"{model_name}-raw-dtr-\d+.json", get_threshold_from_json_file, get_y, 'raw dtr')

    draw_from_file(ax, rf"{model_name}-me-style-div-beta-\d+.json", get_threshold_from_json_file, get_y, 'me div beta')
    draw_from_file(ax, rf"{model_name}-me-style-mul-beta-\d+.json", get_threshold_from_json_file, get_y, 'me mul beta')

    ax.legend()
    plt.savefig(pic_name)

# draw_from_files_and_draw(get_theo_time_from_json_file, f'{model_name}-theo-time.png')
# draw_from_files_and_draw(get_real_time_from_json_file, f'{model_name}-real-time.png')
draw_from_files_and_draw(get_mem_frag_rate_from_json_file, f'{model_name}-mem-frag.png')
draw_from_files_and_draw(get_dataset_time_from_json_file, f'{model_name}-dataset-time.png')

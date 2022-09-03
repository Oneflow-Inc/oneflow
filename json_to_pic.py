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
    fns = list(
        filter(re.compile(fn_pattern).match, (str(x.name) for x in cwd.iterdir()))
    )
    data = []
    for fn in fns:
        threshold = threshold_fn(fn)
        time = time_fn(fn)
        data.append((threshold, time))
    data = sorted(data, key=lambda x: x[0])

    # def f(x):
    #     if len(x) == 0:
    #         return x
    #     minimal = x[0][0]
    #     x = filter(lambda x: x[0] >= 4000 or x[0] == minimal, x)
    #     return map(lambda x: (x[0] / 9000, x[1]), x)
    #
    # data = f(data)
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
        if mem_frag_rate is None:
            mem_frag_rate = 0
        mem_frag_rate = float(mem_frag_rate)
        return mem_frag_rate


def get_dataset_time_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        x = j["dataset time"]
        if x is not None:
            x = float(x)
        return x


def get_threshold_from_json_file(fn):
    with open(fn) as f:
        j = json.load(f)
        return float(j["threshold"][:-2])


fig, ax = plt.subplots()

model_name = sys.argv[1]


def draw_from_files_and_draw(*, xlabel, ylabel, get_y, pic_name, legend_and_fn_pattern):
    _, ax = plt.subplots()
    max_threshold = 0
    for fn_pattern in legend_and_fn_pattern.values():
        fns = list(
            filter(re.compile(fn_pattern).match, (str(x.name) for x in cwd.iterdir()))
        )
        max_threshold = max(max(map(get_threshold_from_json_file, fns)), max_threshold)

    def get_memory_ratio_from_json_file(*args, **kwargs):
        return get_threshold_from_json_file(*args, **kwargs) / max_threshold

    for label, fn_pattern in legend_and_fn_pattern.items():
        draw_from_file(ax, fn_pattern, get_memory_ratio_from_json_file, get_y, label)

    ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    plt.savefig(pic_name)


# draw_from_files_and_draw(get_theo_time_from_json_file, f'{model_name}-theo-time.png')
# draw_from_files_and_draw(get_real_time_from_json_file, f'{model_name}-real-time.png')
# draw_from_files_and_draw(get_mem_frag_rate_from_json_file, f'{model_name}-mem-frag.png')
draw_from_files_and_draw(
    xlabel="Memory Ratio",
    ylabel="Overhead (x)",
    get_y=get_dataset_time_from_json_file,
    pic_name=f"{model_name}-ablation-study.png",
    legend_and_fn_pattern={
        "ours": rf"{model_name}-ours-\d+.json",
        "no op-guided allocation": rf"{model_name}-no-gp-\d+.json",
        "no memory reuse": rf"{model_name}-no-fbip-\d+.json",
        "no layout-aware eviction": rf"{model_name}-me-style-\d+.json",
        # "DTE (Our Impl)": rf"{model_name}-dte-our-impl-\d+.json",
    },
)

draw_from_files_and_draw(
    xlabel="Memory Ratio",
    ylabel="Memory Fragmentation Rate",
    get_y=get_mem_frag_rate_from_json_file,
    pic_name=f"{model_name}-mem-frag.png",
    legend_and_fn_pattern={
        "ours": rf"{model_name}-ours-\d+.json",
        "no op-guided allocation": rf"{model_name}-no-gp-\d+.json",
        "no memory reuse": rf"{model_name}-no-fbip-\d+.json",
        "no layout-aware eviction": rf"{model_name}-me-style-\d+.json",
        "DTE (Our Impl)": rf"{model_name}-dte-our-impl-\d+.json",
        "DTR (Our Impl)": rf"{model_name}-raw-dtr-\d+.json",
    },
)

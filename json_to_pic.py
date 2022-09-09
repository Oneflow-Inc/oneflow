import json
import csv
import sys
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os


our_times = []
nlr_times = []
no_allo_times = []
no_lr_times = []


cwd = Path.cwd()


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
            mem_frag_rate = np.nan
        mem_frag_rate = float(mem_frag_rate)
        # return %
        return mem_frag_rate * 100


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
        t = float(j["threshold"][:-2])
        if t == 9900:
            t = 10000
        return t


model_name = sys.argv[1]


def draw_one(ax, data, label, marker, i, total, kind):
    assert kind in ["step", "line", "bar"]
    zorder = 200 - i
    length = len(data)
    data = list(zip(*data))
    if kind == "step":
        ax.step(
            *data,
            where="post",
            label=label,
            linewidth=4,
            marker=marker,
            markevery=[True] + [False] * (length - 1),
            ms=10,
            zorder=zorder,
        )
    elif kind == "line":
        ax.plot(
            *data,
            label=label,
            linewidth=3,
            marker=marker,
            markevery=1,
            ms=10,
            zorder=zorder,
        )
    elif kind == "bar":
        width = 0.15
        x = np.arange(len(data[0]))
        ax.bar(
            x + (i - total / 2 + 0.5) * width,
            data[1],
            width=width * 0.88,
            label=label,
            zorder=zorder,
        )


def draw_from_files_and_draw(
    *,
    xlabel,
    ylabel,
    get_y,
    pic_name,
    legend_and_fn_patterns,
    ncols,
    nrows,
    kind,
    imgcat=False,
):
    assert len(legend_and_fn_patterns) == ncols * nrows
    fig, axs = plt.subplots(ncols=ncols, nrows=nrows, figsize=(12, 6), sharey=True)

    for i, ax in enumerate(axs):
        _draw_from_files_and_draw_in_subplot(
            ax=ax,
            xlabel=xlabel,
            ylabel=ylabel,
            get_y=get_y,
            legend_and_fn_pattern=legend_and_fn_patterns[i],
            kind=kind,
        )
    left = 0.05
    right = 1
    subplot_center = (left + right) / 2
    fig.subplots_adjust(top=1, left=left, right=right, bottom=0.08, wspace=0.1)
    handles, labels = ax.get_legend_handles_labels() # type: ignore
    fig.legend(handles, labels, loc='upper center', ncol=5, bbox_to_anchor=(subplot_center, 0.01))

    fig.supylabel(ylabel)
    fig.supxlabel(xlabel, x=subplot_center)

    plt.savefig(pic_name, bbox_inches="tight")
    if imgcat:
        os.system(f"imgcat {pic_name}")


def _draw_from_files_and_draw_in_subplot(
    *, ax, xlabel, ylabel, get_y, legend_and_fn_pattern, kind
):
    data = {}
    threshold_set = set()
    for label, fn_pattern in legend_and_fn_pattern.items():
        fns = list(
            filter(re.compile(fn_pattern).match, (str(x.name) for x in cwd.iterdir()))
        )
        thresholds = list(map(get_threshold_from_json_file, fns))
        threshold_set = threshold_set.union(thresholds)
        data[label] = list(zip(thresholds, list(map(get_y, fns))))
    max_threshold = max(threshold_set)
    min_threshold = min(threshold_set)
    if kind == "bar":
        for threshold in threshold_set:
            for label, d in data.items():
                if threshold not in list(zip(*d))[0]:
                    d.append((threshold, np.nan))
    for label in data:
        data[label].sort(key=lambda x: x[0])
        print(data[label])
        data[label] = list(map(lambda x: (x[0] / max_threshold, x[1]), data[label]))
        # pop max_threshold because it doesn't have mem frag
        if kind == "bar":
            data[label].pop()
            threshold_set.discard(max_threshold)
            del data[label][0]
            threshold_set.discard(min_threshold)

    print(f"max_threshold: {max_threshold}")

    marker = ["o", "*", "D", "^", "s"]
    for i, (label, d) in enumerate(data.items()):
        draw_one(ax, d, label, marker[i], i, len(data), kind)

    # ax.set_ylabel(ylabel)
    # ax.set_xlabel(xlabel)
    if kind in ["step", "line"]:
        ax.grid()
    if kind in ["bar"]:
        x = list(map(lambda x: x / max_threshold, sorted(list(threshold_set))))
        ax.set_xticks(np.arange(len(x)), x)
        ax.grid(axis="y")


# draw_from_files_and_draw(get_theo_time_from_json_file, f'{model_name}-theo-time.png')
# draw_from_files_and_draw(get_real_time_from_json_file, f'{model_name}-real-time.png')
# draw_from_files_and_draw(get_mem_frag_rate_from_json_file, f'{model_name}-mem-frag.png')

# draw_from_files_and_draw(
#     xlabel="Memory Ratio",
#     ylabel="Overhead (x)",
#     get_y=get_dataset_time_from_json_file,
#     pic_name=f"{model_name}-ablation-study.png",
#     legend_and_fn_pattern={
#         "ours": rf"{model_name}-ours-\d+.json",
#         "no op-guided allocation": rf"{model_name}-no-gp-\d+.json",
#         "no memory reuse": rf"{model_name}-no-fbip-\d+.json",
#         "no layout-aware eviction": rf"{model_name}-me-style-\d+.json",
#         # "DTE (Our Impl)": rf"{model_name}-dte-our-impl-\d+.json",
#     },
#     imgcat=True,
#     kind=sys.argv[2]
# )
#
# exit()
draw_from_files_and_draw(
    xlabel="Memory Ratio",
    ylabel="Memory Fragmentation Rate (%)",
    get_y=get_mem_frag_rate_from_json_file,
    pic_name=f"{model_name}-mem-frag.png",
    legend_and_fn_patterns=[
        {
            "Ours": rf"{model_name}-ours-\d0000?.json",
            "no op-guided allocation": rf"{model_name}-no-gp-\d0000?.json",
            "no memory reuse": rf"{model_name}-no-fbip-\d0000?.json",
            "no layout-aware eviction": rf"{model_name}-me-style-\d0000?.json",
            "DTE": rf"{model_name}-dte-our-impl-\d0000?.json",
        },
        {
            "Ours": rf"{model_name}-ours-\d0000?.json",
            "no op-guided allocation": rf"{model_name}-no-gp-\d0000?.json",
            "no memory reuse": rf"{model_name}-no-fbip-\d0000?.json",
            "no layout-aware eviction": rf"{model_name}-me-style-\d0000?.json",
            "DTE": rf"{model_name}-dte-our-impl-\d0000?.json",
        },
    ],
    ncols=2,
    nrows=1,
    imgcat=True,
    kind=sys.argv[2],
)


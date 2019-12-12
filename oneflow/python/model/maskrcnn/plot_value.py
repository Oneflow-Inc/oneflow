#  pip3 install -U altair vega_datasets jupyterlab --user
import altair as alt
import pandas as pd
import numpy as np
import os
import glob


def plot_value(df):
    legends = df["legend"].unique()
    poly_data = pd.DataFrame(
        {"iter": np.linspace(df["iter"].min(), df["iter"].max(), 1000)}
    )
    for legend in legends:
        poly_data[legend + "-fit"] = np.poly1d(
            np.polyfit(
                df[df["legend"] == legend]["iter"],
                df[df["legend"] == legend]["value"],
                3,
            )
        )(poly_data["iter"])

    base = alt.Chart(df).interactive()

    chart = base.mark_circle().encode(x="iter", y="value", color="legend:N")

    polynomial_fit = (
        alt.Chart(poly_data)
        .transform_fold(
            [legend + "-fit" for legend in legends], as_=["legend", "value"]
        )
        .mark_line()
        .encode(x="iter:Q", y="value:Q", color="legend:N")
    )
    chart += polynomial_fit
    chart.display()


def plot_by_legend(df):
    legends = df["legend"].unique()
    for legend in legends:
        df[df["legend"] == legend]
        plot_value(df[df["legend"] == legend])


def plot_many_by_legend(df_dict):
    legend_set_unsored = []
    legend_set_sorted = [
        "loss_rpn_box_reg",
        "loss_objectness",
        "lr",
        "loss_box_reg",
        "total_pos_inds_elem_cnt",
        "loss_classifier",
        "loss_mask",
        "elapsed_time",
    ]
    for _, df in df_dict.items():
        for legend in list(df["legend"].unique()):
            if (
                legend not in legend_set_sorted
                and df[df["legend"] == legend]["note"].size is 0
            ):
                legend_set_unsored.append(legend)
    for legend in legend_set_sorted + legend_set_unsored:
        df_by_legend = pd.concat(
            [
                update_legend(df[df["legend"] == legend].copy(), k)
                for k, df in df_dict.items()
            ],
            axis=0,
        )
        plot_value(df_by_legend)


def update_legend(df, prefix):
    if df["legend"].size > 0:
        df["legend"] = df.apply(
            lambda row: "{}-{}".format(prefix, row["legend"]), axis=1
        )
    return df


DL = "/Users/jackal/Downloads/"


COLUMNS = [
    "loss_rpn_box_reg",
    "loss_objectness",
    "loss_box_reg",
    "loss_classifier",
    "loss_mask",
]


def make_loss_frame(hisogram, column_index, legend="undefined"):
    assert column_index < len(COLUMNS)
    ndarray = np.array(hisogram)[:, [column_index, len(COLUMNS)]]
    return pd.DataFrame(
        {"iter": ndarray[:, 1], "legend": legend, "value": ndarray[:, 0]}
    )


def make_loss_frame5(losses_hisogram, source):
    return pd.concat(
        [
            make_loss_frame(losses_hisogram, column_index, legend=column_name)
            for column_index, column_name in enumerate(COLUMNS)
        ],
        axis=0,
        ignore_index=True,
    )


touch_loss_np = np.load(os.path.join(DL, "pytorch_maskrcnn_losses.npy"))

torch_old_70k = make_loss_frame5(touch_loss_np, "pytorch_maskrcnn_losses")

flow = pd.read_csv(
    os.path.join(
        DL,
        "loss-19999-batch_size-8-gpu-4-image_dir-train2017-2019-12-12--02-51-33.csv",
    )
)
torch_before_fix_1x1 = pd.read_csv(
    os.path.join(
        DL,
        "torch-13299-batch_size-8-image_dir-coco_2017_train-2019-12-12--02-01-47.csv",
    )
)


def latest_wildcard(path):
    result = glob.glob(path)
    result.sort(key=os.path.getmtime)
    return result[-1]


flow_latest_csv = latest_wildcard(os.path.join(DL, "loss*.csv"))
torch_latest_csv = latest_wildcard(os.path.join(DL, "torch*.csv"))

flow_latest = pd.read_csv(flow_latest_csv)
torch_latest = pd.read_csv(torch_latest_csv)

LIMIT = min(torch_latest["iter"].max(), flow["iter"].max())
TAKE_EVERY_N = 1
if LIMIT > 2500:
    TAKE_EVERY_N = LIMIT // 2500 + 1


def isinstance2(x, t):
    print(type(x))
    return isinstance(x, t)


def subset_and_mod(df, limit, take_every_n):
    df_limited = df[df["iter"] < limit]
    return df_limited[df_limited["iter"] % take_every_n == 0]


plot_many_by_legend(
    {
        "flow": subset_and_mod(flow_latest, LIMIT, TAKE_EVERY_N),
        "torch": subset_and_mod(torch_latest, LIMIT, TAKE_EVERY_N),
        # "torch_before_fix_1x1": subset_and_mod(
        #     torch_before_fix_1x1, LIMIT, TAKE_EVERY_N
        # ),
        # "torch_old_70k": subset_and_mod(torch_old_70k, LIMIT, TAKE_EVERY_N),
    }
)

#  pip3 install -U altair vega_datasets jupyterlab --user
import altair as alt
import pandas as pd
import numpy as np
import os


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
                5,
            )
        )(poly_data["iter"])

    base = alt.Chart(df).interactive()

    chart = base.mark_line().encode(x="iter", y="value", color="legend:N")

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
    legend_set = []
    for _, df in df_dict.items():
        for legend in list(df["legend"].unique()):
            if legend not in legend_set:
                legend_set.append(legend)
    legend_set = [
        "loss_rpn_box_reg",
        "loss_objectness",
        "loss_box_reg",
        "total_pos_inds_elem_cnt",
        "loss_classifier",
        "loss_mask",
        "elapsed_time",
    ]
    for legend in legend_set:
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

pytorch_maskrcnn_losses = make_loss_frame5(
    touch_loss_np, "pytorch_maskrcnn_losses"
)

flow = pd.read_csv(
    os.path.join(
        DL,
        "init-loss-1999-batch_size-2-gpu-1-image_dir-train2017-2019-12-10--21-05-17.csv",
    )
)
torch = pd.read_csv(
    os.path.join(
        DL,
        "torch-1999-batch_size-2-image_dir-coco_2017_train-2019-12-10--13-07-48.csv",
    )
)

plot_many_by_legend(
    {
        # "torch_lr_0": torch_lr_0[torch_lr_0["iter"] < 100][
        #     torch_lr_0["iter"] % 1 == 0
        # ],
        "flow": flow[flow["iter"] < 2000][flow["iter"] % 2 == 0],
        "torch": torch[torch["iter"] < 2000][torch["iter"] % 2 == 0],
    }
)

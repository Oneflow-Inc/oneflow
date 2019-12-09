import altair as alt
import pandas as pd
import numpy as np

COLUMNS = [
    "loss_rpn_box_reg",
    "loss_objectness",
    "loss_box_reg",
    "loss_classifier",
    "loss_mask",
]


def plot_loss(df):
    legends = df["legend"].unique()
    poly_data = pd.DataFrame(
        {"iter": np.linspace(df["iter"].min(), df["iter"].max(), 1000)}
    )
    for legend in legends:
        poly_data[legend + "-fit"] = np.poly1d(
            np.polyfit(
                df[df["legend"] == legend]["iter"],
                df[df["legend"] == legend]["loss"],
                2,
            )
        )(poly_data["iter"])

    base = alt.Chart(df).interactive()

    chart = base.mark_line().encode(x="iter", y="loss", color="legend:N")

    polynomial_fit = (
        alt.Chart(poly_data)
        .transform_fold(
            [legend + "-fit" for legend in legends], as_=["legend", "loss"]
        )
        .mark_line()
        .encode(x="iter:Q", y="loss:Q", color="legend:N")
    )
    chart += polynomial_fit
    chart.display()

    if len(legends) == 2:
        loss_ratio_df = pd.DataFrame(
            {
                "metrics": poly_data[legends[0] + "-fit"]
                / poly_data[legends[1] + "-fit"],
                "iter": poly_data["iter"],
                "legend": "loss_ratio",
            }
        )
        loss_diff_df = pd.DataFrame(
            {
                "metrics": poly_data[legends[0] + "-fit"]
                - poly_data[legends[1] + "-fit"],
                "iter": poly_data["iter"],
                "legend": "loss_diff",
            }
        )
        loss_compare_df = pd.concat([loss_ratio_df, loss_diff_df], axis=0)
        base = alt.Chart(loss_compare_df).interactive()
        chart = (
            base.mark_line()
            .encode(x="iter:Q", y="metrics:Q", color="legend:N")
            .mark_line()
        )
        chart.display()


def plot_lr(df):
    base = alt.Chart(df).interactive()

    chart = base.mark_line().encode(x="iter:Q", y="lr:Q", color="legend:N")

    chart.display()


def take_every_n(ndarray, n):
    return ndarray[np.mod(np.arange(ndarray.shape[0]), n) == 0]


def make_loss_frame(hisogram, column_index, legend="undefined"):
    assert column_index < len(COLUMNS)
    ndarray = np.array(hisogram)[:, [column_index, len(COLUMNS)]]
    return pd.DataFrame(
        {"legend": legend, "loss": ndarray[:, 0], "iter": ndarray[:, 1]}
    )


def make_lr_frame(hisogram, legend="undefined"):
    if hisogram.shape[1] == 7:
        ndarray = np.array(hisogram)[:, [6, 5]]
        return pd.DataFrame(
            {"legend": legend, "lr": ndarray[:, 0], "iter": ndarray[:, 1]}
        )
    else:
        return pd.DataFrame()


def plot5(losses_hisogram, source="undefined"):
    for column_index, column_name in enumerate(COLUMNS):
        plot_loss(make_loss_frame(losses_hisogram, column_index, column_name))


def plot5in1(losses_hisogram, source="undefined"):
    cat = pd.concat(
        [
            make_loss_frame(
                losses_hisogram,
                column_index,
                legend="{}-{}".format(column_name, source),
            )
            for column_index, column_name in enumerate(COLUMNS)
        ],
        axis=0,
    )
    plot_loss(cat)


def plot2(losses_hisogram_dict):
    cat_lr = pd.concat(
        [
            make_lr_frame(losses_hisogram, legend="{}-{}".format("lr-", source))
            for source, losses_hisogram in losses_hisogram_dict.items()
        ],
        axis=0,
    )
    plot_lr(cat_lr)
    for column_index, column_name in enumerate(COLUMNS):
        cat = pd.concat(
            [
                make_loss_frame(
                    losses_hisogram,
                    column_index,
                    legend="{}-{}".format(column_name, source),
                )
                for source, losses_hisogram in losses_hisogram_dict.items()
            ],
            axis=0,
        )
        plot_loss(cat)


plot2(
    {
        "OneFlow-sunday": take_every_n(
            np.load(
                "/Users/jackal/Downloads/loss-19999-batch_size-8-gpu-4-image_dir-train2017-2019-12-09--01-02-54.npy"
            )[:6000],
            3,
        ),
        "PyTorch-no_contigous": take_every_n(
            np.load(
                "/Users/jackal/Downloads/pytorch_maskrcnn_losses_no_contigous.npy"
            )[:6000],
            3,
        ),
    }
)
# plot2(
#     {
#         "PyTorch-no_contigous": take_every_n(np.load("/Users/jackal/Downloads/pytorch_maskrcnn_losses_no_contigous.npy")[:6000],3),
#         "PyTorch": take_every_n(np.load("/Users/jackal/Downloads/pytorch_maskrcnn_losses.npy")[:6000], 3)
#     }
# )
# plot2(
#     {
#         "OneFlow-sunday": np.load("/Users/jackal/Downloads/loss-1999-batch_size-8-gpu-4-image_dir-train2017-2019-12-08--18-55-06.npy")[:1600, :],
#         "OneFlow-friday": np.load("/Users/jackal/Downloads/loss-1999-batch_size-8-gpu-4-image_dir-train2017.npy")[:1600, :],
#     }
# )
# plot5in1(take_every_n(np.load("/tmp/shared_with_zwx/pytorch_maskrcnn_losses.npy"), 100), "PyTorch")
# plot5(take_every_n(np.load("/Users/jackal/Downloads/pytorch_maskrcnn_losses.npy"), 100), "PyTorch")

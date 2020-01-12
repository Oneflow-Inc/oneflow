import pandas as pd
import sqlite3
import altair as alt
from plot_value import plot_value


def set_pd_opts():
    pd.set_option("display.max_rows", 1500)
    pd.options.display.max_colwidth = 100


def rank(con):
    # unit should be 1 ns
    df = pd.read_sql_query(
        "SELECT text, (end - start) AS duration FROM NVTX_EVENTS;", con
    )
    df = df[~df["text"].str.contains("Kernel::Launch")]
    # df = df[~df["text"].str.contains("sync")]
    grouped = (
        df.groupby("text")
        .agg({"duration": ["median", "mean", "std", "min", "max", "count"]})
        .sort_values(by=[("duration", "median")], ascending=False)
    )

    print(grouped[:100])


ACTIVITY_KIND = ["KERNEL", "MEMCPY", "MEMSET", "RUNTIME", "SYNCHRONIZATION"]


def query_kernel_activity_with_runtime_span(con, r_start, r_end):
    sql = f"""
    SELECT k.correlationId, r.start as r_start, r.end as r_end, k.start, k.end, s.value as name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME as r
            LEFT JOIN CUPTI_ACTIVITY_KIND_KERNEL as k
                JOIN StringIds as s
    ON r.correlationId = k.correlationId and s.id = k.demangledName
    where r.start >= {r_start}
    and r.end <= {r_end}
    """
    return pd.read_sql_query(sql, con)


def query_activity_with_runtime_span(con, r_start, r_end, kind):
    kind_str = kind.upper()
    assert kind_str in ACTIVITY_KIND
    sql = f"""
    SELECT k.correlationId, r.start as r_start, r.end as r_end, k.start, k.end
    FROM CUPTI_ACTIVITY_KIND_RUNTIME as r
            LEFT JOIN CUPTI_ACTIVITY_KIND_{kind_str} as k
    ON r.correlationId = k.correlationId
    where r.start >= {r_start}
    and r.end <= {r_end}
    and k.correlationId not null
    """
    return pd.read_sql_query(sql, con)


def query_all_activity_with_runtime_span(con, r_start, r_end):
    return pd.concat(
        [
            query_activity_with_runtime_span(con, r_start, r_end, kind)
            for kind in ACTIVITY_KIND
            if kind is not "RUNTIME"
        ],
        axis=0,
    )


def query_activity(con, kind):
    kind_str = kind.upper()
    assert kind_str in ACTIVITY_KIND
    sql = f"""
    SELECT a.correlationId, r.start as runtime_start, r.end as runtime_end, a.start as gpu_start, a.end as gpu_end
    FROM CUPTI_ACTIVITY_KIND_RUNTIME as r 
        LEFT JOIN CUPTI_ACTIVITY_KIND_{kind_str} as a
    ON r.correlationId = a.correlationId
    where a.correlationId not null and r.correlationId not null
    """
    df = pd.read_sql_query(sql, con)
    df["kind"] = kind
    return df


def query_all_activity(con):
    return pd.concat(
        [
            query_activity(con, kind)
            for kind in ACTIVITY_KIND
            if kind is not "RUNTIME"
        ],
        axis=0,
    )


def query_nvtx_with_text(con, text):
    df = pd.read_sql_query(
        f"SELECT start, end FROM NVTX_EVENTS where text = '{text}';", con
    )
    return df


def get_gpu_span_with_text_by_query(con, text):
    nvtx_df = query_nvtx_with_text(con, text)

    def update_gpu_span(nvtx):
        activity = query_all_activity_with_runtime_span(
            con, nvtx.start, nvtx.end
        )
        start_min = activity.start.min()
        end_max = activity.end.max()
        nvtx["gpu_span"] = end_max - start_min
        return nvtx

    return nvtx_df.apply(update_gpu_span, axis=1)


def get_gpu_span_with_text(con, text, activity_df):
    nvtx_df = query_nvtx_with_text(con, text)

    def update_gpu_span(nvtx):
        activity = activity_df[
            (activity_df.runtime_start > nvtx.start)
            & (activity_df.runtime_end < nvtx.end)
        ]
        start_min = activity.gpu_start.min()
        end_max = activity.gpu_end.max()

        nvtx["gpu_end"] = end_max
        nvtx["gpu_span"] = end_max - start_min
        return nvtx

    nvtx_df["text"] = text
    nvtx_df["iter"] = nvtx_df.index + 1
    return nvtx_df.apply(update_gpu_span, axis=1)


def get_gpu_span(con, text_list):
    if isinstance(text_list, (str)):
        text_list = [text_list]
    else:
        assert isinstance(text_list, (list, tuple))
    activity_df = query_all_activity(con)
    return pd.concat(
        [get_gpu_span_with_text(con, text, activity_df) for text in text_list],
        axis=0,
        sort=False,
    )


def test1():
    db_path = "file:///home/tsai/Downloads/torch-2020-01-10-06:41:12-UTC.sqlite"
    # 46813
    # 1st bw
    # nvtx
    # 63482843839 63577992899
    # in runtime
    # 63577960546 63577977294
    # in kernel
    # 63634936998 63634938342
    con = sqlite3.connect(db_path)
    bw_span = query_nvtx_with_text(con, "Backward pass").iloc[0]
    print(query_all_activity_with_runtime_span(con, bw_span.start, bw_span.end))
    con.close()


if __name__ == "__main__":
    set_pd_opts()
    db_path = "file:///home/tsai/Downloads/1x-perf-2/mask-rcnn-unknown-2020-01-09-17:08:15-CST.sqlite"
    db_path = "/Users/jackal/Downloads/mask-rcnn-unknown-2020-01-10-23-50-15-CST.sqlite"
    db_path = "/Users/jackal/Downloads/maskrcnn.sqlite"
    con = sqlite3.connect(db_path)
    text = "Backward pass"
    # df = get_gpu_span(con, ["Forward pass", "Backward pass"])
    text_list = [
        "backbone",
        "rpn_head",
        "rpn_loss",
        "rpn_post_processor",
        "box_head",
        "mask_head",
    ]
    df = get_gpu_span(con, text_list)
    df["value"] = df["gpu_span"] / 1e6
    df["legend"] = df["text"]
    df = df[df["iter"] > 1]
    plot_value(df)
    # print(df)
    con.close()

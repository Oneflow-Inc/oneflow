import pandas as pd
import sqlite3


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


ACTIVITY_KIND = [
    "KERNEL",
    "MEMCPY",
    "MEMSET",
    "RUNTIME",
    "SYNCHRONIZATION",
]


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
    ).sort_values(by=["end"], ascending=True)


def query_span_with_text(con, text):
    df = pd.read_sql_query(
        f"SELECT start, end FROM NVTX_EVENTS where text = '{text}';", con
    )
    return df


# debug with file:///home/tsai/Downloads/torch-2020-01-10-06:41:12-UTC.sqlite
# 46813
# 1st bw
# 63482843839 63577992899
# in runtime
# 63577960546 63577977294
if __name__ == "__main__":
    set_pd_opts()
    db_path = "file:///home/tsai/Downloads/1x-perf-2/mask-rcnn-unknown-2020-01-09-17:08:15-CST.sqlite"
    db_path = "file:///home/tsai/Downloads/torch-2020-01-10-06:41:12-UTC.sqlite"
    con = sqlite3.connect(db_path)
    bw_span = query_span_with_text(con, "Backward pass").iloc[0]
    print(query_all_activity_with_runtime_span(con, bw_span.start, bw_span.end))
    con.close()

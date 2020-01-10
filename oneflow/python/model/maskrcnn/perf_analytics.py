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


def query_activity_of_runtime_activity(r_start, r_end, kind):
    kind_str = kind.upper()
    sql = f"""
    SELECT k.correlationId, r.start as r_start, r.end as r_end, k.start, k.end, s.value as name
    FROM CUPTI_ACTIVITY_KIND_RUNTIME as r
            LEFT JOIN CUPTI_ACTIVITY_KIND_{kind_str} as k
                JOIN StringIds as s
    ON r.correlationId = k.correlationId and s.id = k.demangledName
    where r.start >= {r_start}
    and r.end <= {r_end}
    """


if __name__ == "__main__":
    set_pd_opts()
    con = sqlite3.connect(
        "file:///home/tsai/Downloads/1x-perf-2/mask-rcnn-unknown-2020-01-09-17:08:15-CST.sqlite"
    )
    rank(con)
    con.close()

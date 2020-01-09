import pandas as pd
import sqlite3

pd.set_option("display.max_rows", 1500)
pd.options.display.max_colwidth = 100
con = sqlite3.connect(
    "file:///home/tsai/Downloads/1x-perf-2/mask-rcnn-unknown-2020-01-09-17:08:15-CST.sqlite"
)
# unit should be 1 ns
df = pd.read_sql_query("SELECT text, (end - start) AS duration FROM NVTX_EVENTS;", con)
df = df[~df["text"].str.contains("Kernel::Launch")]
grouped = (
    df.groupby("text")
    .agg({"duration": ["median", "mean", "std", "min", "max", "count"]})
    .sort_values(by=[("duration", "std")], ascending=False)
)

print(grouped[:100])

con.close()

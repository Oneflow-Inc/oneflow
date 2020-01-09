import pandas as pd
import sqlite3

pd.set_option("display.max_rows", 1500)
pd.options.display.max_colwidth = 100
con = sqlite3.connect(
    "file:///home/tsai/Downloads/mask-rcnn-unknown-1x-2020-01-08-16:23:46-CST.sqlite"
)
# unit should be 1 ns
df = pd.read_sql_query("SELECT text, (end - start) AS duration FROM NVTX_EVENTS;", con)
grouped = (
    df.groupby("text")
    .agg({"duration": ["median", "mean", "std", "min", "max", "count"]})
    .sort_values(by=[("duration", "median")], ascending=False)
)

print(grouped[:1000])

con.close()

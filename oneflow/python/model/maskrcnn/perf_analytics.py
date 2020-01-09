import pandas as pd
import sqlite3


con = sqlite3.connect("file:///home/tsai/Downloads/mask-rcnn-unknown-1x-2020-01-08-16:23:46-CST.sqlite")
df = pd.read_sql_query("SELECT text, (end - start) AS duration FROM NVTX_EVENTS;", con)
grouped = df.groupby("text")
print(grouped.)

con.close()

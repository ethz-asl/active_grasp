#%%
import pandas as pd

#%%
logfile = "../logs/210712-132211_policy=top.csv"
df = pd.read_csv(logfile)

#%%
metrics = [
    ("Runs", len(df.index)),
    ("", ""),
    ("Succeeded", (df.result == "succeeded").sum()),
    ("Failed", (df.result == "failed").sum()),
    ("Aborted", (df.result == "aborted").sum()),
    ("", ""),
    ("Success rate", round((df.result == "succeeded").mean(), 2)),
    ("Mean time", round(df.exploration_time.mean(), 2)),
    ("Mean distance", round(df.distance_travelled.mean(), 2)),
    ("Mean viewpoints", round(df.viewpoint_count.mean())),
]

for k, v in metrics:
    print("{:<16} {:>8}".format(k, v))

# %%

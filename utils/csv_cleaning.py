import pandas as pd

df = pd.read_csv("data/metadata.csv")
df = pd.add_below()
df.to_csv("data/metadata.csv", index=False)

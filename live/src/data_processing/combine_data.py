import os
import pandas as pd
from pathlib import Path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))

OUTPUT_DIR = Path('../output/hmm')

df_1 = pd.read_csv(os.path.join(root_dir, "output", "hmm", "hmm_regimes.csv"))
df_2 = pd.read_csv(os.path.join(root_dir, "output", "hmm_data.csv"))
print('Loading data success')

res = pd.merge_asof(
    df_1,
    df_2,
    on="time"
)

print(res.head(5))

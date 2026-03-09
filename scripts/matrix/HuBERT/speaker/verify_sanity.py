#!/usr/bin/env python3
# compare_self_vs_selfdouble.py
import pandas as pd
import numpy as np
from pathlib import Path

csv_path = Path("/gpfs/projects/bsc88/speech/mm_s2st/repos/paraling_speech/src/outputs/matrices/siamese_matrix/siamese_matrix.csv")
out_csv = csv_path.parent / "auc_diff_sorted_fixed.csv"

df = pd.read_csv(csv_path)

# Normalize lang_tgt: treat empty string as NaN
df['lang_tgt'] = df['lang_tgt'].replace("", np.nan)

# Build a 'key' column:
#  - base rows: key = lang_src
#  - dual rows: key = lang_src_lang_tgt  (keep the actual lang_tgt)
df['key'] = df.apply(lambda r: r['lang_src'] if pd.isna(r['lang_tgt']) else f"{r['lang_src']}_{r['lang_tgt']}", axis=1)

# Aggregate auc_mean by key (in case there are multiple rows per key)
agg = df.groupby('key', as_index=True)['auc_mean'].mean()

results = []
languages = sorted(df['lang_src'].unique())

for lang in languages:
    base_key = lang
    doubled_key = f"{lang}_{lang}"

    auc_base = agg.get(base_key, np.nan)
    auc_doubled = agg.get(doubled_key, np.nan)

    if np.isnan(auc_base) or np.isnan(auc_doubled):
        # skip if either missing
        continue

    diff = auc_doubled - auc_base
    pct = diff / auc_base if auc_base != 0 else np.nan
    if diff > 0:
        flag = "improved"
    elif diff == 0:
        flag = "no_change"
    else:
        flag = "degraded"

    results.append({
        "lang": lang,
        "auc_base": float(auc_base),
        "auc_doubled": float(auc_doubled),
        "diff": float(diff),
        "pct_change": float(pct),
        "status": flag
    })

res_df = pd.DataFrame(results)
res_df = res_df.sort_values(by="diff", ascending=False).reset_index(drop=True)

# Print to stdout (pretty)
pd.set_option("display.width", 120)
pd.set_option("display.precision", 6)
print(res_df.to_string(index=False))

# Save CSV
res_df.to_csv(out_csv, index=False)
print(f"\nSaved sorted results to: {out_csv}")

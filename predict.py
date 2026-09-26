"""Stage 4: score test candidates, apply the tuned decision rule and write the
two submission files (every test S1 entity gets exactly one row)."""
import json

import lightgbm as lgb
import polars as pl

from config import DATA_DIR, WORK_DIR, OUT_DIR
from features import FEATURES
from train import decide

model = lgb.Booster(model_file=str(WORK_DIR / "matcher.txt"))
dec = json.load(open(WORK_DIR / "decision.json"))
ctx = pl.scan_parquet(WORK_DIR / "ctx_test.parquet")
preds = []
for p in sorted(WORK_DIR.glob("rowf_test_*.parquet")):
    x = pl.read_parquet(p)
    lo, hi = x["s1_id"].min(), x["s1_id"].max()   # parts are sorted by s1_id
    cx = ctx.filter(pl.col("s1_id").is_between(pl.lit(lo), pl.lit(hi))).collect()
    x = x.join(cx, on=["s1_id", "o_id"], how="left")
    del cx
    preds.append(x.select("s1_id", "o_id").with_columns(
        pl.Series("p", model.predict(x.select(pl.col(FEATURES).cast(pl.Float32)).to_numpy()))))
pred = pl.concat(preds)
pred.write_parquet(WORK_DIR / "test_pred.parquet")

s1 = pl.read_csv(DATA_DIR / "test" / "test_source1.tsv", separator="\t", quote_char=None,
                 infer_schema=False, columns=["entity_id"]).rename({"entity_id": "s1_id"})


def write(pairs: pl.DataFrame, col: str, path):
    agg = pairs.unique(["s1_id", "o_id"]).sort(["s1_id", "o_id"]).group_by("s1_id").agg(pl.col("o_id").str.join(",").alias(col))
    out = s1.join(agg, on="s1_id", how="left").with_columns(pl.col(col).fill_null(""))
    out = out.rename({"s1_id": "source1_entity_id"})
    # plain tab-separated text, no quoting, empty cell for no match
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{col}\n")
        for a, b in out.iter_rows():
            f.write(f"{a}\t{b}\n")
    return out


cand = write(pred.select("s1_id", "o_id"), "candidate_entity_ids", OUT_DIR / "candidate_pairs.tsv")
match = write(decide(pred, dec["threshold"], dec["assign"]), "matched_entity_ids", OUT_DIR / "matching_results.tsv")
n_match = (match["matched_entity_ids"] != "").sum()
print(f"S1 rows {s1.height}; with >=1 match {n_match}; matched ids "
      f"{decide(pred, dec['threshold'], dec['assign']).height}; candidates {pred.height}")

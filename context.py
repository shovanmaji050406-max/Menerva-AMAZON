"""Stage 2b: context features over the full candidate table (memory-lean).
Reads only the few columns needed from the row-feature parts, computes the
competition features, and writes work/ctx_<split>.parquet keyed by (s1_id, o_id)."""
import sys
import polars as pl
from config import WORK_DIR
from features import add_context, CONTEXT_FEATURES

split = sys.argv[1]
parts = sorted(WORK_DIR.glob(f"rowf_{split}_*.parquet"))
lean = pl.concat([pl.read_parquet(p, columns=["s1_id", "o_id", "blk_score", "n_tset", "a_tset"]) for p in parts])
lean = lean.with_columns(pl.col("s1_id").cast(pl.Categorical), pl.col("o_id").cast(pl.Categorical))
ctx = add_context(lean.lazy()).select(["s1_id", "o_id"] + CONTEXT_FEATURES).collect()
ctx = ctx.with_columns(pl.col("s1_id").cast(pl.String), pl.col("o_id").cast(pl.String),
                       *[pl.col(c).cast(pl.Float32) for c in CONTEXT_FEATURES])
ctx.write_parquet(WORK_DIR / f"ctx_{split}.parquet")
print("ctx rows", ctx.height)

"""Stage 2 driver: row-wise pair features in chunks (bounded memory), then
context features over the full candidate table.  Output: work/feats_<split>.parquet"""
import sys, time
import polars as pl
from config import DATA_DIR, WORK_DIR
from features import record_table, pair_features, add_context, FEATURES

CH = 1_500_000
split = sys.argv[1]
D = DATA_DIR / split
sc = lambda f: pl.scan_csv(D / f, separator="\t", quote_char=None, infer_schema=False)
cands = pl.concat([pl.read_parquet(p) for p in sorted(WORK_DIR.glob(f"cands_{split}_*.parquet"))]).sort("s1_id")
print("pairs", cands.height, flush=True)
S1 = sc(f"{split}_source1.tsv")
OT = pl.concat([sc(f"{split}_source2.tsv"), sc(f"{split}_source3.tsv")])
parts = []
for i in range(0, cands.height, CH):
    t = time.time()
    c = cands.slice(i, CH)
    r1 = record_table(S1.join(c.select(pl.col("s1_id").alias("entity_id")).unique().lazy(), on="entity_id", how="semi").collect())
    r2 = record_table(OT.join(c.select(pl.col("o_id").alias("entity_id")).unique().lazy(), on="entity_id", how="semi").collect())
    f = pair_features(c, r1, r2)
    fp = WORK_DIR / f"rowf_{split}_{i // CH:03d}.parquet"
    f.write_parquet(fp); parts.append(fp)
    print(f"chunk {i // CH}: {c.height} pairs {time.time()-t:.0f}s", flush=True)
x = add_context(pl.scan_parquet(parts)).select(["s1_id", "o_id"] + FEATURES)
x.sink_parquet(WORK_DIR / f"feats_{split}.parquet")
for p in parts: p.unlink()
print("done", flush=True)

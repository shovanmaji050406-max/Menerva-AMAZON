"""Stage 1: candidate generation for a split (train/test), all countries.
Writes work/cands_<split>.parquet with blocking scores per (s1_id, o_id)."""
import sys, time
import polars as pl
from blocking import candidates_for_country
from config import DATA_DIR, WORK_DIR, POOL_M, POOL_K

split = sys.argv[1]
D = DATA_DIR / split
sc = lambda f: pl.scan_csv(D / f, separator="\t", quote_char=None, infer_schema=False)
countries = sc(f"{split}_source1.tsv").select("country").unique().collect()["country"].drop_nulls().to_list()
print(split, "countries:", countries, flush=True)
outs = []
for c in countries:
    t = time.time()
    s1 = sc(f"{split}_source1.tsv").filter(pl.col("country") == c).collect()
    oth = pl.concat([sc(f"{split}_source2.tsv"), sc(f"{split}_source3.tsv")]).filter(pl.col("country") == c).collect()
    p = candidates_for_country(s1, oth, top_m_per_other=POOL_M, top_k_per_s1=POOL_K)
    p = p.with_columns(pl.lit(c).alias("country"))
    p.write_parquet(WORK_DIR / f"cands_{split}_{c}.parquet")
    print(f"{c}: s1={s1.height} oth={oth.height} pairs={p.height} ({time.time()-t:.0f}s)", flush=True)

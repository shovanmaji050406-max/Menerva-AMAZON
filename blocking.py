"""Scalable candidate generation (blocking).

Idea: every record emits a handful of *keys*:
  * N:<skeleton>          one per informative name token (transliteration/typo tolerant)
  * A:<number>|<word5>    house/plot number paired with each informative address
                          word (first 5 chars) -- very specific, order independent
Keys are weighted by inverse document frequency within the country.  Very
common keys are dropped (they would explode the join and carry no signal).
Source-1 records and Source-2/3 records sharing keys become pairs, scored by the
summed IDF of shared keys.

Two pruning passes keep candidate sets small (Amazon ranks smaller candidate
sets higher):
  1. per Source-2/3 record keep only its top-M Source-1 anchors (each S2/S3
     record belongs to at most one S1 entity in the ground truth), and
  2. per Source-1 anchor keep its top-K candidates.
Everything is done per country, so countries never mix and unseen countries
(France) are handled identically.
"""
import math

import shutil, tempfile
from pathlib import Path

import polars as pl
import resource, time

def _log(msg):
    print(f"  [{time.strftime('%H:%M:%S')}] {msg}  maxrss={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6:.2f}GB", flush=True)

from textnorm import add_clean_columns, name_tokens, addr_parts

MAX_DF_S1 = 20         # drop keys shared by more S1 records than this
MAX_DF_OTH = 80       # ... or by more S2/S3 records than this
MAX_WORDS = 4
MAX_NUMS = 3
MAX_NAME_TOK = 4
CHUNK = 250_000
JOIN_CHUNKS = 24
N_PART = 8
TMP_DIR = str(Path(__file__).resolve().parents[1] / "work")


def _spill(df: pl.DataFrame, out: Path):
    """Build keys chunk by chunk and write them hash-partitioned to parquet."""
    out.mkdir(parents=True, exist_ok=True)
    for ci, i in enumerate(range(0, df.height, CHUNK)):
        k = _build_keys(df.slice(i, CHUNK)).with_columns((pl.col("h") % N_PART).alias("part"))
        for (p,), g in k.group_by("part"):
            g.drop("part").write_parquet(out / f"p{p}_{ci:04d}.parquet")
    _log(f"spilled {out.name}")


def build_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Chunked wrapper so peak memory stays low; returns (rid:u32, h:u64, kind:u8)."""
    parts = [_build_keys(df.slice(i, CHUNK)) for i in range(0, df.height, CHUNK)]
    return pl.concat(parts)


def _pairs(t: pl.DataFrame, col: str) -> pl.DataFrame:
    """unordered within-record pairs of a token column -> (rid, a, b)."""
    return (t.select("rid", pl.col(col).alias("a")).join(t.select("rid", pl.col(col).alias("b")), on="rid")
            .filter(pl.col("a") < pl.col("b")))


def _build_keys(df: pl.DataFrame) -> pl.DataFrame:
    """Conjunctive keys. Single names / single address parts are NOT specific in
    this data (the same business name occurs at dozens of addresses and the same
    building hosts many businesses), so every key combines two or three pieces."""
    df = add_clean_columns(df.select("rid", "business_name", "business_address"))
    nt = name_tokens(df)
    nt = nt.with_columns(pl.int_range(pl.len()).over("rid").alias("i")).filter(pl.col("i") < MAX_NAME_TOK)
    nums, words = addr_parts(df)
    nums = nums.with_columns(pl.int_range(pl.len()).over("rid").alias("i")).filter(pl.col("i") < MAX_NUMS)
    words = (words.with_columns(pl.col("w").str.slice(0, 6))
             .unique(maintain_order=True)
             .with_columns(pl.int_range(pl.len()).over("rid").alias("i"))
             .filter(pl.col("i") < MAX_WORDS))
    glued = (nt.group_by("rid", maintain_order=True).agg(pl.col("sk").str.join("").alias("g"))
             .filter(pl.col("g").str.len_chars() >= 4)
             .with_columns(pl.col("g").str.slice(0, 8)))
    np_ = _pairs(nt, "sk")                       # name token pairs
    wp = _pairs(words, "w")                      # address word pairs
    n_ = nums.select("rid", "n")
    w_ = words.select("rid", "w")
    t_ = nt.select("rid", "sk")
    K = []
    # name-pair x number
    K.append(np_.join(n_, on="rid").select("rid", ("a" + pl.col("a") + "|" + pl.col("b") + "|" + pl.col("n")).alias("k")))
    # name token x number
    K.append(t_.join(n_, on="rid").select("rid", ("b" + pl.col("sk") + "|" + pl.col("n")).alias("k")))
    # glued name x number, glued name x word (domains / hashtags)
    K.append(glued.join(n_, on="rid").select("rid", ("c" + pl.col("g") + "|" + pl.col("n")).alias("k")))
    K.append(glued.join(w_, on="rid").select("rid", ("d" + pl.col("g") + "|" + pl.col("w")).alias("k")))
    # number x word pair (junk / non-latin names)
    K.append(n_.join(wp, on="rid").select("rid", ("f" + pl.col("n") + "|" + pl.col("a") + "|" + pl.col("b")).alias("k")))
    # number x word
    K.append(n_.join(w_, on="rid").select("rid", ("g" + pl.col("n") + "|" + pl.col("w")).alias("k")))
    # number pair (e.g. '45/2763', '6-3-251')
    K.append(_pairs(nums, "n").select("rid", ("h" + pl.col("a") + "|" + pl.col("b")).alias("k")))
    # name token x word
    K.append(t_.join(w_, on="rid").select("rid", ("j" + pl.col("sk") + "|" + pl.col("w")).alias("k")))
    # NAME-ONLY keys (records with empty / useless address): full name skeleton
    # and name-token pairs.  Only survive the frequency filter for rare names.
    full = (nt.group_by("rid", maintain_order=True).agg(pl.col("sk").sort().str.join("|").alias("g"))
            .filter(pl.col("g").str.len_chars() >= 4))
    K.append(full.select("rid", ("k" + pl.col("g")).alias("k")))
    K.append(np_.select("rid", ("l" + pl.col("a") + "|" + pl.col("b")).alias("k")))
    k = pl.concat(K).unique()
    return k.select(
        pl.col("rid").cast(pl.UInt32),
        pl.col("k").hash(seed=7).alias("h"),
        pl.col("k").str.slice(0, 1).is_in(["a", "b", "c", "d", "e", "i", "j", "k", "l"]).cast(pl.UInt8).alias("kind"),  # 1=uses name
    )


def candidates_for_country(s1: pl.DataFrame, oth: pl.DataFrame,
                           top_m_per_other: int = 2, top_k_per_s1: int = 8, debug_s1=None) -> pl.DataFrame:
    """s1/oth: DataFrames with entity_id, business_name, business_address.
    Returns pairs (s1_id, o_id, blk_score, blk_withname, blk_addronly, blk_nshared,
    rank_in_s1, rank_in_o)."""
    n1 = s1.height
    s1 = s1.with_row_index("rid")
    oth = oth.with_row_index("rid").with_columns(pl.col("rid") + n1)
    ids1 = s1.select("rid", pl.col("entity_id").alias("s1_id"))
    ids2 = oth.select(pl.col("rid").alias("orid"), pl.col("entity_id").alias("o_id"))
    # --- keys are spilled to disk, hash-partitioned, so memory stays flat ---
    tmp = Path(tempfile.mkdtemp(prefix="blk_", dir=TMP_DIR))
    _spill(s1, tmp / "s1")
    n2 = oth.height
    _spill(oth, tmp / "o")
    del s1, oth
    N = n1 + n2
    k1_parts, k2_parts = [], []
    for p in range(N_PART):
        k1 = pl.read_parquet(tmp / "s1" / f"p{p}_*.parquet")
        d1 = k1.group_by("h").agg(pl.len().alias("d1")).filter(pl.col("d1") <= MAX_DF_S1)
        k2 = pl.read_parquet(tmp / "o" / f"p{p}_*.parquet").join(d1.select("h"), on="h", how="semi")
        d2 = k2.group_by("h").agg(pl.len().alias("d2")).filter(pl.col("d2") <= MAX_DF_OTH)
        dfs = (d1.join(d2, on="h")
               .with_columns((N / (pl.col("d1") + pl.col("d2"))).log().cast(pl.Float32).alias("idf"))
               .select("h", "idf"))
        k1_parts.append(k1.join(dfs, on="h").select("rid", "h", "kind", "idf"))
        k2_parts.append(k2.join(dfs.select("h"), on="h", how="semi").select(pl.col("rid").alias("orid"), "h"))
        del k1, k2, d1, d2, dfs
    shutil.rmtree(tmp, ignore_errors=True)
    k1 = pl.concat(k1_parts); k2 = pl.concat(k2_parts)
    del k1_parts, k2_parts
    _log(f"filtered keys k1={k1.height} k2={k2.height}")
    # join in chunks of S2/S3 records; each chunk is fully aggregated and
    # pruned to top-M anchors per S2/S3 record before moving on.
    lo, hi = n1, n1 + n2
    step = max(1, (hi - lo) // JOIN_CHUNKS + 1)
    out = []
    for a in range(lo, hi, step):
        kc = k2.filter(pl.col("orid").is_between(a, a + step - 1))
        pc = (k1.join(kc, on="h")
              .group_by("rid", "orid")
              .agg(pl.col("idf").sum().alias("blk_score"),
                   pl.col("idf").filter(pl.col("kind") == 1).sum().alias("blk_withname"),
                   pl.col("idf").filter(pl.col("kind") == 0).sum().alias("blk_addronly"),
                   pl.len().cast(pl.UInt16).alias("blk_nshared")))
        pc = pc.with_columns(
            pl.col("blk_score").rank("min", descending=True).over("orid").cast(pl.UInt16).alias("rank_in_o"))
        if debug_s1 is not None:
            dbg = ids1.filter(pl.col("s1_id") == debug_s1)["rid"][0]
            print("DEBUG", pc.filter(pl.col("rid") == dbg))
        out.append(pc.filter(pl.col("rank_in_o") <= top_m_per_other))
        _log(f"chunk {a} pairs {pc.height}")
    del k1, k2
    pairs = pl.concat(out)
    del out
    # prune: each anchor keeps its best top-K candidates
    pairs = pairs.with_columns(
        pl.col("blk_score").rank("ordinal", descending=True).over("rid").alias("rank_in_s1"))
    pairs = pairs.filter(pl.col("rank_in_s1") <= top_k_per_s1)
    # relative score features
    pairs = pairs.with_columns(
        (pl.col("blk_score") / pl.col("blk_score").max().over("orid")).alias("blk_rel_o"),
        (pl.col("blk_score") / pl.col("blk_score").max().over("rid")).alias("blk_rel_s1"),
    )
    pairs = (pairs.join(ids1.with_columns(pl.col("rid").cast(pl.UInt32)), on="rid")
             .join(ids2.with_columns(pl.col("orid").cast(pl.UInt32)), on="orid")
             .drop("rid", "orid"))
    return pairs

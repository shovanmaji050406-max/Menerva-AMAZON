"""Stage 2: pair features for candidate pairs.

Features are country-agnostic (no country one-hot), so the model transfers to
the unseen test country (France).
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist

from textnorm import add_clean_columns, name_tokens, addr_parts, NAME_STOP


def record_table(df: pl.DataFrame) -> pl.DataFrame:
    """Per-record normalised views used by the pair features."""
    df = df.with_row_index("rid")
    c = add_clean_columns(df.select("rid", "business_name", "business_address"))
    nt = name_tokens(c).group_by("rid").agg(pl.col("sk").alias("sk_list"))
    nums, words = addr_parts(c)
    nums = nums.group_by("rid", maintain_order=True).agg(pl.col("n").alias("num_list"))
    words = words.group_by("rid").agg(pl.col("w").alias("w_list"))
    stop_re = r"\b(" + "|".join(sorted(NAME_STOP - {"&"}, key=len, reverse=True)) + r")\b"
    c = c.with_columns(
        pl.col("name_c").str.replace_all(stop_re, " ").str.replace_all(r"\s+", " ").str.strip_chars().alias("core"),
        pl.col("business_name").fill_null("").str.contains(r"[^\x00-\x7F]").alias("nonascii"),
        pl.col("business_name").fill_null("").str.contains(r"(?i)\.(com|in|net|org|co|fr)\b|^#|www\.").alias("webby"),
        pl.col("business_address").is_null().alias("addr_null"),
    )
    out = (df.select("rid", "entity_id").join(c.select("rid", "name_c", "addr_c", "core", "nonascii", "webby", "addr_null"), on="rid")
           .join(nt, on="rid", how="left").join(nums, on="rid", how="left").join(words, on="rid", how="left")
           .drop("rid"))
    out = out.with_columns(
        pl.col("sk_list").fill_null([]), pl.col("num_list").fill_null([]), pl.col("w_list").fill_null([]),
        pl.col("sk_list").list.join("").fill_null("").alias("glued"),
    )
    return out


def _jacc(a: str, b: str) -> pl.Expr:
    inter = pl.col(a).list.set_intersection(pl.col(b)).list.len()
    union = pl.col(a).list.set_union(pl.col(b)).list.len()
    return pl.when(union > 0).then(inter / union).otherwise(0.0)


def name_idf_table(s1: pl.DataFrame) -> pl.DataFrame:
    """IDF of name skeleton tokens among Source-1 records of one split, per country.
    Common words ('group', 'india', 'trading') get low weight, distinctive
    words get high weight."""
    s1 = s1.with_row_index("rid")
    c = add_clean_columns(s1.select("rid", "business_name", "business_address"))
    t = name_tokens(c).join(s1.select("rid", "country"), on="rid")
    n = s1.group_by("country").agg(pl.len().alias("N"))
    return (t.group_by("country", "sk").agg(pl.col("rid").n_unique().alias("df")).join(n, on="country")
            .select("country", "sk", (pl.col("N") / pl.col("df")).log().cast(pl.Float32).alias("idf")))


def _idf_feats(x: pl.DataFrame, idf: pl.DataFrame) -> pl.DataFrame:
    """Name-rarity features: how much *distinctive* name evidence is shared."""
    base = x.select(pl.int_range(pl.len()).alias("pi"), "country", "a_sk_list", "b_sk_list")
    maxidf = idf["idf"].max()

    def total(col):
        e = base.select("pi", "country", pl.col(col).alias("sk")).explode("sk").drop_nulls("sk")
        e = e.join(idf, on=["country", "sk"], how="left").with_columns(pl.col("idf").fill_null(maxidf))
        return e.group_by("pi").agg(pl.col("idf").sum().alias(col + "_w"), pl.col("idf").max().alias(col + "_max"))

    sh = base.select("pi", "country", pl.col("a_sk_list").list.set_intersection(pl.col("b_sk_list")).alias("sk")).explode("sk").drop_nulls("sk")
    sh = sh.join(idf, on=["country", "sk"], how="left").with_columns(pl.col("idf").fill_null(maxidf))
    sh = sh.group_by("pi").agg(pl.col("idf").sum().alias("sh_w"), pl.col("idf").max().alias("sh_max"))
    f = base.select("pi").join(total("a_sk_list"), on="pi", how="left").join(total("b_sk_list"), on="pi", how="left").join(sh, on="pi", how="left").fill_null(0)
    f = f.sort("pi").select(
        pl.col("sh_w").alias("name_shared_idf"),
        pl.col("sh_max").alias("name_shared_maxidf"),
        (pl.col("sh_w") / pl.col("a_sk_list_w").clip(lower_bound=1e-3)).alias("name_idf_cov_a"),
        (pl.col("sh_w") / pl.col("b_sk_list_w").clip(lower_bound=1e-3)).alias("name_idf_cov_b"),
        (pl.col("a_sk_list_max") - pl.col("sh_max")).alias("name_unshared_rare_a"),
    )
    return pl.concat([x, f], how="horizontal")


def pair_features(pairs: pl.DataFrame, rec1: pl.DataFrame, rec2: pl.DataFrame, idf: pl.DataFrame = None) -> pl.DataFrame:
    """pairs: s1_id, o_id, country + blocking columns.  rec1/rec2 from record_table()."""
    r1 = rec1.rename({c: "a_" + c for c in rec1.columns if c != "entity_id"}).rename({"entity_id": "s1_id"})
    r2 = rec2.rename({c: "b_" + c for c in rec2.columns if c != "entity_id"}).rename({"entity_id": "o_id"})
    x = pairs.join(r1, on="s1_id", how="left").join(r2, on="o_id", how="left")

    def cd(a, b, scorer, **kw):
        return cpdist(x[a].fill_null("").to_list(), x[b].fill_null("").to_list(),
                      scorer=scorer, workers=-1, dtype=np.float32, **kw)

    feats = {
        "n_ratio": cd("a_name_c", "b_name_c", fuzz.ratio),
        "n_tset": cd("a_name_c", "b_name_c", fuzz.token_set_ratio),
        "n_tsort": cd("a_name_c", "b_name_c", fuzz.token_sort_ratio),
        "n_partial": cd("a_name_c", "b_name_c", fuzz.partial_ratio),
        "core_tset": cd("a_core", "b_core", fuzz.token_set_ratio),
        "core_ratio": cd("a_core", "b_core", fuzz.ratio),
        "glued_ratio": cd("a_glued", "b_glued", fuzz.ratio),
        "glued_partial": cd("a_glued", "b_glued", fuzz.partial_ratio),
        "glued_jw": cd("a_glued", "b_glued", JaroWinkler.normalized_similarity),
        "a_ratio": cd("a_addr_c", "b_addr_c", fuzz.ratio),
        "a_tset": cd("a_addr_c", "b_addr_c", fuzz.token_set_ratio),
        "a_tsort": cd("a_addr_c", "b_addr_c", fuzz.token_sort_ratio),
        "a_partial": cd("a_addr_c", "b_addr_c", fuzz.partial_ratio),
    }
    x = x.with_columns([pl.Series(k, v) for k, v in feats.items()])
    x = x.with_columns(
        _jacc("a_sk_list", "b_sk_list").alias("sk_jacc"),
        pl.col("a_sk_list").list.set_intersection(pl.col("b_sk_list")).list.len().alias("sk_inter"),
        _jacc("a_num_list", "b_num_list").alias("num_jacc"),
        pl.col("a_num_list").list.set_intersection(pl.col("b_num_list")).list.len().alias("num_inter"),
        (pl.col("b_num_list").list.set_difference(pl.col("a_num_list")).list.len()).alias("num_b_extra"),
        (pl.col("a_num_list").list.set_difference(pl.col("b_num_list")).list.len()).alias("num_a_extra"),
        _jacc("a_w_list", "b_w_list").alias("w_jacc"),
        (pl.col("a_num_list").list.first() == pl.col("b_num_list").list.first()).fill_null(False).alias("num_first_eq"),
        (pl.col("a_num_list").list.max().cast(pl.Int64, strict=False) ==
         pl.col("b_num_list").list.max().cast(pl.Int64, strict=False)).fill_null(False).alias("num_max_eq"),
        pl.col("a_sk_list").list.len().alias("a_ntok"),
        pl.col("b_sk_list").list.len().alias("b_ntok"),
        pl.col("b_num_list").list.len().alias("b_nnum"),
        pl.col("a_num_list").list.len().alias("a_nnum"),
        pl.col("b_addr_c").str.len_chars().alias("b_addr_len"),
        pl.col("a_addr_c").str.len_chars().alias("a_addr_len"),
        pl.col("b_nonascii").cast(pl.Int8), pl.col("b_webby").cast(pl.Int8), pl.col("b_addr_null").cast(pl.Int8),
        pl.col("o_id").str.starts_with("S3").cast(pl.Int8).alias("is_s3"),
        ((pl.col("a_num_list").list.len() > 0) & (pl.col("b_num_list").list.len() > 0)
         & (pl.col("a_num_list").list.first() != pl.col("b_num_list").list.first())).fill_null(False).cast(pl.Int8).alias("num_first_conflict"),
        ((pl.col("a_num_list").list.len() > 0) & (pl.col("b_num_list").list.len() > 0)
         & (pl.col("a_num_list").list.set_intersection(pl.col("b_num_list")).list.len() == 0)).cast(pl.Int8).alias("num_no_overlap"),
    )
    x = _idf_feats(x, idf)
    return x.select(["s1_id", "o_id"] + ROW_FEATURES)


def add_context(x: pl.LazyFrame) -> pl.LazyFrame:
    """Competition / context features over the FULL candidate table: how does a
    pair compare with the other anchors of the same S2/S3 record and the other
    candidates of the same S1 anchor?"""
    return x.with_columns(
        pl.len().over("s1_id").alias("s1_ncand"),
        pl.len().over("o_id").alias("o_nanch"),
        (pl.col("blk_score") - pl.col("blk_score").max().over("o_id")).alias("blk_gap_o"),
        (pl.col("n_tset") - pl.col("n_tset").max().over("o_id")).alias("ntset_gap_o"),
        (pl.col("a_tset") - pl.col("a_tset").max().over("o_id")).alias("atset_gap_o"),
        (pl.col("n_tset") - pl.col("n_tset").mean().over("s1_id")).alias("ntset_vs_s1mean"),
        (pl.col("a_tset") - pl.col("a_tset").mean().over("s1_id")).alias("atset_vs_s1mean"),
        (pl.col("n_tset") + pl.col("a_tset")).alias("_sum"),
    ).with_columns(
        (pl.col("_sum") - pl.col("_sum").max().over("o_id")).alias("sum_gap_o"),
        (pl.col("_sum") - pl.col("_sum").max().over("s1_id")).alias("sum_gap_s1"),
        pl.col("_sum").rank("min", descending=True).over("o_id").alias("sum_rank_o"),
    ).drop("_sum")


ROW_FEATURES = [
    "blk_score", "blk_withname", "blk_addronly", "blk_nshared", "rank_in_o", "rank_in_s1",
    "blk_rel_o", "blk_rel_s1",
    "n_ratio", "n_tset", "n_tsort", "n_partial", "core_tset", "core_ratio",
    "glued_ratio", "glued_partial", "glued_jw",
    "a_ratio", "a_tset", "a_tsort", "a_partial",
    "sk_jacc", "sk_inter", "num_jacc", "num_inter", "num_b_extra", "num_a_extra", "w_jacc",
    "num_first_eq", "num_max_eq", "a_ntok", "b_ntok", "b_nnum", "a_nnum", "b_addr_len", "a_addr_len",
    "b_nonascii", "b_webby", "b_addr_null", "is_s3",
    "num_first_conflict", "num_no_overlap",
    "name_shared_idf", "name_shared_maxidf", "name_idf_cov_a", "name_idf_cov_b", "name_unshared_rare_a",
]
CONTEXT_FEATURES = ["s1_ncand", "o_nanch", "blk_gap_o", "ntset_gap_o", "atset_gap_o",
                    "ntset_vs_s1mean", "atset_vs_s1mean", "sum_gap_o", "sum_gap_s1", "sum_rank_o"]
FEATURES = ROW_FEATURES + CONTEXT_FEATURES

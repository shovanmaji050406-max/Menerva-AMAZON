"""Text normalisation and token/key generation for business entity resolution.

All heavy string work is done with vectorised Polars expressions; only the
per-*unique-token* canonicalisation (skeleton) runs in Python, which keeps it
fast even for ~10M records.
"""
import re
from functools import lru_cache

import polars as pl
from anyascii import anyascii

# ---------------------------------------------------------------------------
# Stop-lists (country agnostic: US / India / France + transliterated forms)
# ---------------------------------------------------------------------------
NAME_STOP = {
    # legal forms
    "pvt", "private", "prvt", "ltd", "limited", "llc", "llp", "inc", "incorporated",
    "corp", "corporation", "co", "company", "pc", "pllc", "lp", "plc", "public",
    "sarl", "sas", "sasu", "eurl", "sa", "sci", "ei", "snc", "selarl", "gmbh",
    # generic filler words that the noise process adds / drops
    "the", "and", "of", "et", "fils", "de", "du", "des", "la", "le", "les", "l", "d",
    "com", "www", "http", "https", "net", "org", "in", "services", "service",
    "group", "enterprises", "enterprise", "partners", "holdings", "ta", "dba",
    "pra", "li", "a", "an", "for", "&",
}

ADDR_STOP = {
    "road", "rd", "street", "st", "avenue", "ave", "av", "boulevard", "blvd", "bd",
    "drive", "dr", "lane", "ln", "highway", "hwy", "apartment", "apt", "appt",
    "suite", "ste", "building", "bldg", "floor", "fl", "flr", "unit", "no", "nr",
    "plot", "near", "opp", "opposite", "behind", "house", "h", "door", "block",
    "sector", "phase", "road", "marg", "nagar", "colony", "court", "ct", "circle",
    "cir", "place", "pl", "way", "rue", "r", "allee", "all", "impasse", "imp",
    "chemin", "ch", "quai", "route", "rte", "cours", "square", "sq", "de", "du",
    "des", "la", "le", "les", "d", "l", "et", "bis", "ter", "the", "and", "of",
    "null", "n", "a", "po", "dist", "district", "taluk", "tq", "ward", "main",
    "cross", "west", "east", "north", "south", "w", "e", "s", "floor", "ground",
    "first", "second", "third", "1st", "2nd", "3rd", "4th", "township", "cdp",
    "city", "town", "village", "post", "office", "kh", "khasra", "gali", "hn",
}

_DIGIT_FIX = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s",
                            "6": "g", "8": "b", "9": "g", "7": "t", "2": "z"})


@lru_cache(maxsize=None)
def skeleton(tok: str) -> str:
    """Phonetic-ish consonant skeleton, robust to typos, accents and Indic
    transliteration (e.g. 'modern'/'modrn' -> 'mdrn', 'food'/'phud' -> 'ft',
    'limited'/'limitet' -> 'lmtt').  Mixed OCR tokens like 'micr0' -> 'mkr'."""
    if not tok:
        return tok
    has_d = any(c.isdigit() for c in tok)
    has_a = any(c.isalpha() for c in tok)
    if has_d and not has_a:
        return "#" + (tok.lstrip("0") or "0")
    if has_d and has_a:
        tok = tok.translate(_DIGIT_FIX)
    t = tok.replace("ph", "f").replace("ck", "k")
    t = t.translate(str.maketrans({"c": "k", "q": "k", "z": "s", "j": "g", "d": "t",
                                   "v": "b", "x": "k"}))
    t = re.sub(r"[aeiouyhw]", "", t)
    t = re.sub(r"(.)\1+", r"\1", t)
    return t or tok


STOP_SKEL = {skeleton(w) for w in NAME_STOP if len(w) > 2} | {
    skeleton(w) for w in ("praivet", "limitet", "privet", "limted", "pvtltd")}


def to_ascii_lower(col: str) -> pl.Expr:
    """anyascii() only for non-ASCII strings (Indic scripts, accents), then lowercase."""
    c = pl.col(col).fill_null("")
    return (
        pl.when(c.str.contains(r"[^\x00-\x7F]"))
        .then(c.map_elements(anyascii, return_dtype=pl.String))
        .otherwise(c)
        .str.to_lowercase()
    )


def add_clean_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Adds name_c / addr_c : ascii, lowercase, punctuation -> space."""
    df = df.with_columns(
        to_ascii_lower("business_name").alias("name_c"),
        to_ascii_lower("business_address").alias("addr_c"),
    )
    df = df.with_columns(
        pl.col("name_c")
        .str.replace_all(r"\.(com|in|net|org|co|fr)\b", " ")
        .str.replace_all(r"&", " and ")
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.strip_chars(),
        pl.col("addr_c")
        .str.replace_all(r"<null>|\bn/?a\b", " ")
        .str.replace_all(r"[^a-z0-9]+", " ")
        .str.strip_chars(),
    )
    return df


def name_tokens(df: pl.DataFrame, id_col="rid") -> pl.DataFrame:
    """Long table (rid, tok) of skeletonised name tokens (stop words removed)."""
    t = (df.select(id_col, pl.col("name_c").str.split(" ").alias("tok"))
         .explode("tok").filter(pl.col("tok").str.len_chars() > 0)
         .filter(~pl.col("tok").is_in(list(NAME_STOP))))
    uniq = t.select("tok").unique()
    uniq = uniq.with_columns(pl.col("tok").map_elements(skeleton, return_dtype=pl.String).alias("sk"))
    t = t.join(uniq, on="tok", how="left", maintain_order="left").filter(~pl.col("sk").is_in(list(STOP_SKEL)))
    t = t.filter(pl.col("sk").str.len_chars() >= 2)
    return t.select(id_col, "sk").unique(maintain_order=True)


def addr_parts(df: pl.DataFrame, id_col="rid"):
    """Returns (nums, words) long tables for the address: numbers with leading
    zeros stripped, and informative alpha words (>=3 chars, not generic)."""
    nums = (df.select(id_col, pl.col("addr_c").str.extract_all(r"\d+").alias("n"))
            .explode("n").drop_nulls()
            .with_columns(pl.col("n").str.strip_chars_start("0"))
            .filter(pl.col("n").str.len_chars().is_between(1, 6)).unique(maintain_order=True))
    words = (df.select(id_col, pl.col("addr_c").str.extract_all(r"[a-z]{3,}").alias("w"))
             .explode("w").drop_nulls()
             .filter(~pl.col("w").is_in(list(ADDR_STOP))).unique(maintain_order=True))
    return nums, words

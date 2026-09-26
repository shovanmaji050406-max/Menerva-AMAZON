"""Stage 3: train the pairwise matcher (LightGBM) and tune the decision rule for
macro F0.5 on held-out Source-1 entities.

Validation is done per Source-1 entity exactly like the leaderboard: every
held-out S1 entity counts (including ones with no candidates at all, and true
singletons, which score 1.0 only if we predict nothing)."""
import json
import pickle

import lightgbm as lgb
import numpy as np
import polars as pl

from config import DATA_DIR, WORK_DIR
from features import FEATURES

VAL_FRAC = 0.15
SEED = 42


def load_truth():
    gt = pl.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", separator="\t", infer_schema=False)
    pairs = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
             .explode("matched_entity_ids")
             .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "o_id"})
             .filter(pl.col("o_id").is_not_null() & (pl.col("o_id") != "")))
    n_true = pairs.group_by("s1_id").agg(pl.len().alias("n_true"))
    all_s1 = gt.select(pl.col("source1_entity_id").alias("s1_id"))
    return all_s1, pairs.with_columns(pl.lit(1, pl.Int8).alias("label")), n_true


def is_val(col="s1_id") -> pl.Expr:
    return (pl.col(col).hash(seed=SEED) % 1000) < int(VAL_FRAC * 1000)


def decide(pred: pl.DataFrame, thr: float, assign: bool = True) -> pl.DataFrame:
    """pred: s1_id, o_id, p.  Each S2/S3 record is given to at most one S1
    anchor (its highest-probability one) and only if p >= thr."""
    x = pred
    if assign:
        x = x.filter(pl.col("p") == pl.col("p").max().over("o_id"))
    return x.filter(pl.col("p") >= thr).select("s1_id", "o_id")


def macro_f05(matches: pl.DataFrame, s1_ids: pl.DataFrame, truth_pairs: pl.DataFrame, n_true: pl.DataFrame) -> float:
    tp = matches.join(truth_pairs.select("s1_id", "o_id"), on=["s1_id", "o_id"]).group_by("s1_id").agg(pl.len().alias("tp"))
    npred = matches.group_by("s1_id").agg(pl.len().alias("n_pred"))
    t = (s1_ids.join(n_true, on="s1_id", how="left").join(npred, on="s1_id", how="left").join(tp, on="s1_id", how="left")
         .fill_null(0))
    t = t.with_columns(
        pl.when(pl.col("n_pred") > 0).then(pl.col("tp") / pl.col("n_pred")).otherwise(0.0).alias("P"),
        pl.when(pl.col("n_true") > 0).then(pl.col("tp") / pl.col("n_true")).otherwise(0.0).alias("R"))
    f = (pl.when((pl.col("n_true") == 0) & (pl.col("n_pred") == 0)).then(1.0)
         .when(pl.col("tp") == 0).then(0.0)
         .otherwise(1.25 * pl.col("P") * pl.col("R") / (0.25 * pl.col("P") + pl.col("R"))))
    return t.select(f.mean()).item()


TRAIN_SAMPLE = (pl.col("s1_id").hash(seed=SEED) % 1000) >= 850      # 15% of S1 entities (v2)


def load_split(filt, truth, keep_ids=False):
    """Row + context features of the S1 entities selected by `filt`, as float32 numpy."""
    ctx = pl.scan_parquet(WORK_DIR / "ctx_train.parquet")
    Xs, ys, ids = [], [], []
    for p in sorted(WORK_DIR.glob("rowf_train_*.parquet")):
        x = pl.read_parquet(p).filter(filt)
        if x.height == 0:
            continue
        lo, hi = x["s1_id"].min(), x["s1_id"].max()
        x = x.join(ctx.filter(pl.col("s1_id").is_between(pl.lit(lo), pl.lit(hi))).collect(), on=["s1_id", "o_id"], how="left")
        x = x.join(truth.select("s1_id", "o_id", "label"), on=["s1_id", "o_id"], how="left").with_columns(pl.col("label").fill_null(0))
        Xs.append(x.select(pl.col(FEATURES).cast(pl.Float32)).to_numpy()); ys.append(x["label"].to_numpy())
        if keep_ids:
            ids.append(x.select("s1_id", "o_id", "label"))
    return np.vstack(Xs), np.concatenate(ys), (pl.concat(ids) if keep_ids else None)


def main():
    all_s1, truth, n_true = load_truth()
    Xtr, ytr, _ = load_split(TRAIN_SAMPLE, truth)
    Xva, yva, va = load_split(is_val(), truth, keep_ids=True)
    print(f"train pairs {len(ytr)} (pos {ytr.sum()}), val pairs {len(yva)}", flush=True)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  num_threads=2, verbose=-1, seed=SEED)
    dtr = lgb.Dataset(Xtr, label=ytr, feature_name=FEATURES, free_raw_data=True)
    dva = lgb.Dataset(Xva, label=yva, reference=dtr)
    model = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(40), lgb.log_evaluation(50)])
    del Xtr, dtr
    va = va.with_columns(pl.Series("p", model.predict(Xva)))

    val_s1 = all_s1.filter(is_val())
    best = (0, None, None)
    for assign in (True, False):
        for thr in np.arange(0.2, 0.96, 0.05):
            f = macro_f05(decide(va, thr, assign), val_s1, truth, n_true)
            if f > best[0]:
                best = (f, float(thr), assign)
            print(f"assign={assign} thr={thr:.2f} macroF0.5={f:.4f}")
    print("BEST", best)
    # candidate recall ceiling on val
    cand_hits = va.filter(pl.col("label") == 1).height
    tot = truth.filter(is_val()).height
    print(f"val candidate recall ceiling {cand_hits/tot:.4f}; cands/S1 {va.height/val_s1.height:.2f}")
    imp = sorted(zip(FEATURES, model.feature_importance("gain")), key=lambda t: -t[1])
    print("top features:", [(f, round(g)) for f, g in imp[:15]])
    model.save_model(str(WORK_DIR / "matcher.txt"))
    json.dump({"threshold": best[1], "assign": best[2], "val_macro_f05": best[0]},
              open(WORK_DIR / "decision.json", "w"), indent=2)
    va.select("s1_id", "o_id", "label", "p").write_parquet(WORK_DIR / "val_pred.parquet")


if __name__ == "__main__":
    main()

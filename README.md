# Business Entity Resolution — v1 pipeline

Produces `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Validation (held-out 15% of train S1 entities): **macro F0.5 = 0.946**.

## Layout
```
dataset/train/*.tsv, dataset/test/*.tsv   <- official data (not included)
src/                                      <- code
work/                                     <- intermediate files (created)
output/                                   <- submission files (created)
```

## Run (Python 3.11, `pip install -r requirements.txt`), from `src/`
| # | Command | What it does |
|---|---------|--------------|
| 1 | `python run_blocking.py train` and `python run_blocking.py test` | Candidate generation (blocking) per country |
| 2 | `python build_features.py train` and `... test` | 40 row-wise similarity features per pair (writes `work/rowf_*`; the final "context" step inside it can be ignored if it runs out of memory) |
| 3 | `python context.py train` and `python context.py test` | 10 competition/context features over the full candidate table |
| 4 | `python train.py` | Trains LightGBM, tunes threshold for macro F0.5, saves `work/matcher.txt`, `work/decision.json` |
| 5 | `python predict.py` | Scores test pairs, writes both output TSVs |
| 6 | `python ../utils/validate_submission.py -m ../output/matching_results.tsv -c ../output/candidate_pairs.tsv -t ../dataset/test --check-ids` | Official validator |

Tested on a 2-CPU / 7 GB RAM Linux machine (~45 min end to end).

## Method (short)
1. **Normalisation** (`textnorm.py`): transliterate non-Latin scripts (anyascii), lowercase, strip punctuation; name tokens -> consonant "skeleton" (e.g. black/ब्लैक -> `blk`, trading/ट्रेडिंग -> `trtng`) robust to typos/transliteration; address numbers & informative words.
2. **Blocking** (`blocking.py`): conjunctive keys (name-pair x house number, name x number, number x street-word pair, glued-name x number, ...), IDF-weighted, rare keys only, per country. Each S2/S3 record keeps its top-2 S1 anchors, each S1 keeps its top-10 candidates. Keys are spilled to disk hash-partitioned to keep memory flat.
3. **Features** (`features.py`): rapidfuzz name/address similarities, skeleton-token Jaccard, number agreement, webby/non-ASCII flags, and competition features (gap to the best anchor of the same S2/S3 record).
4. **Model** (`train.py`): LightGBM (MIT licence), binary match classifier.
5. **Decision**: each S2/S3 record is assigned to at most one S1 (its highest-probability anchor) and only if p >= 0.60 (tuned for macro F0.5).

# Business Entity Resolution — v2 pipeline

Produces `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
Validation (held-out 15% of train S1 entities): **macro F0.5 = 0.9512** (v1: 0.9462).

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
| 1 | `python run_blocking.py train` ; `python run_blocking.py test` | Candidate generation (blocking) per country |
| 2 | `python build_features.py train` ; `python build_features.py test` | Row-wise similarity features per pair (`work/rowf_*`) |
| 3 | `python context.py train` ; `python context.py test` | Competition/context features over the full candidate table |
| 4 | `python train.py` | LightGBM (15% train / 15% validation S1 entities), threshold tuned for macro F0.5 |
| 5 | `python predict.py` | Scores test pairs, writes both output TSVs |
| 6 | `python ../utils/validate_submission.py -m ../output/matching_results.tsv -c ../output/candidate_pairs.tsv -t ../dataset/test --check-ids` | Official validator |

Tested on a 2-CPU / 7 GB RAM Linux machine (~45 min end to end).

## Method
1. **Normalisation** (`textnorm.py`): anyascii transliteration, lowercase, punctuation removal; name tokens -> consonant skeleton (black/ब्लैक -> `blk`).
2. **Blocking** (`blocking.py`): IDF-weighted *conjunctive* keys, per country:
   name-pair x house number, name x number, glued-name x number/word, number x street-word(-pair), number pairs, name x street word,
   and **name-only keys** (full name skeleton, name-token pairs) so records with an empty address can still be matched.
   Keys shared by > 20 S1 or > 80 S2/S3 records are dropped. Each S2/S3 record keeps its top-2 S1 anchors, each S1 its top-10 candidates.
3. **Features** (`features.py`): rapidfuzz name/address similarities, skeleton-token Jaccard, number agreement,
   **house-number conflict**, **name rarity (IDF-weighted shared name tokens)**, webby/non-ASCII flags, and competition features.
4. **Model**: LightGBM (MIT licence).
5. **Decision**: each S2/S3 record assigned to at most one S1 (its highest-probability anchor) if p >= 0.65.

## v1 -> v2 changes
| Change | Why |
|---|---|
| Name-only blocking keys | 46% of v1's blocking misses had an empty S2/S3 address |
| Looser key frequency caps (12/45 -> 20/80) | recover pruned true pairs |
| Name-rarity features | wrong merges at shared addresses ("Herrera Green Group" vs "Robinson Group Green") |
| House-number conflict features | wrong merges in the same street/building (18404 vs 18413) |
Result: candidate recall 90.3% -> 92.2%, validation F0.5 0.9462 -> 0.9512.

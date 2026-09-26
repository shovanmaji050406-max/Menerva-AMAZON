from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"                   # dataset/{train,test}/*.tsv
WORK_DIR = ROOT / "work"; WORK_DIR.mkdir(exist_ok=True)
OUT_DIR = ROOT / "output"; OUT_DIR.mkdir(exist_ok=True)
POOL_M = 2      # each S2/S3 record keeps its best POOL_M Source-1 anchors
POOL_K = 10     # each Source-1 anchor keeps its best POOL_K candidates

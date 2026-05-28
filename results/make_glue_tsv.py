"""
Convert CATS SST-2 test predictions to GLUE submission format.

GLUE SST-2 expects: SST-2.tsv with tab-separated `index` and `prediction`
columns (0 = negative, 1 = positive).  The TSV and a zip of it are written
next to the input predictions.csv.

Usage:
    python results/make_glue_tsv.py
    python results/make_glue_tsv.py --predictions <path/to/predictions.csv>
"""

import argparse
import zipfile
from pathlib import Path

import pandas as pd


DEFAULT_PREDICTIONS = (
    Path(__file__).parent
    / "core_experiments/full_routing/sst2/cats/carson/run_001/eval_test_best/predictions.csv"
)


def make_glue_tsv(predictions_path: Path) -> Path:
    df = pd.read_csv(predictions_path)

    submission = pd.DataFrame({
        "index": df["sample_index"].astype(int),
        "prediction": df["predicted_label"].astype(int),
    })

    tsv_path = predictions_path.parent / "SST-2.tsv"
    submission.to_csv(tsv_path, sep="\t", index=False)
    print(f"Saved {len(submission)} predictions -> {tsv_path}")

    zip_path = predictions_path.parent / "SST-2.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(tsv_path, arcname="SST-2.tsv")
    print(f"Zipped -> {zip_path}")

    return tsv_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        default=DEFAULT_PREDICTIONS,
        help="Path to predictions.csv produced by evaluate.py",
    )
    args = parser.parse_args()
    make_glue_tsv(args.predictions)

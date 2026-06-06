"""
1. Remove any "All Experiments Summary" cells from all notebooks.
2. Insert a simple save+display cell after the setup cell (id=513501bd)
   in each individual train_log_analysis_*.ipynb notebook.
Run from the project root.
"""
import json, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MARKER     = "All Experiments Summary"
SETUP_CELL = "513501bd"   # the cell that loads df = pd.read_csv(CSV_PATH)

NEW_CELL_SOURCE = """\
# Save train_log.csv to results/ (logs/ is git-ignored)
_save_path = OUT_DIR / 'train_log.csv'
df.to_csv(_save_path, index=False)
print(f'Saved  ->  {_save_path}')
display(df)\
"""


def make_code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


notebooks = sorted(ROOT.glob("results/core_experiments_analysis/**/*.ipynb"))
print(f"Found {len(notebooks)} notebooks\n")

for nb_path in notebooks:
    with open(nb_path, encoding="utf-8") as f:
        nb = json.load(f)

    # Step 1: remove any existing "All Experiments Summary" cells
    before = len(nb["cells"])
    nb["cells"] = [
        c for c in nb["cells"]
        if MARKER not in (c["source"] if isinstance(c["source"], str) else "".join(c["source"]))
    ]
    removed = before - len(nb["cells"])

    # Step 2: for individual notebooks only, insert save+display cell after setup cell
    inserted = False
    if nb_path.name.startswith("train_log_analysis_"):
        # check it doesn't already have the cell
        already = any(
            "OUT_DIR / 'train_log.csv'" in (c["source"] if isinstance(c["source"], str) else "".join(c["source"]))
            for c in nb["cells"]
        )
        if not already:
            for i, c in enumerate(nb["cells"]):
                if c.get("id") == SETUP_CELL:
                    nb["cells"].insert(i + 1, make_code_cell(NEW_CELL_SOURCE))
                    inserted = True
                    break

    if removed or inserted:
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
            f.write("\n")
        tag = []
        if removed:  tag.append(f"removed {removed} summary cell(s)")
        if inserted: tag.append("inserted save+display cell")
        print(f"  UPDATED [{', '.join(tag)}]: {nb_path.relative_to(ROOT)}")
    else:
        print(f"  OK (no changes): {nb_path.name}")

print("\nDone.")

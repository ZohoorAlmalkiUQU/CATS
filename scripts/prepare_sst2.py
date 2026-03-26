from pathlib import Path
import pandas as pd
from datasets import load_dataset


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def preview_split(name, split, n=3):
    print(f"\n--- {name.upper()} SPLIT ---")
    print(f"Number of samples: {len(split)}")
    for i in range(min(n, len(split))):
        example = split[i]
        print(f"\nExample {i + 1}:")
        print(f"Sentence: {example['sentence']}")
        if 'label' in example:
            print(f"Label: {example['label']}")


def save_split_to_csv(split, output_path: Path) -> None:
    df = pd.DataFrame(split)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def main():
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw" / "sst2"
    ensure_dir(raw_dir)

    print("Loading SST-2 from HuggingFace...")
    dataset = load_dataset("glue", "sst2")

    print("\nAvailable splits:")
    for split_name in dataset.keys():
        print(f"- {split_name}: {len(dataset[split_name])} samples")

    # Preview splits
    for split_name in dataset.keys():
        preview_split(split_name, dataset[split_name], n=3)

    # Save raw CSV files locally
    for split_name in dataset.keys():
        output_file = raw_dir / f"{split_name}.csv"
        save_split_to_csv(dataset[split_name], output_file)

    print("\nSST-2 preparation completed successfully.")


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, output_path: Path) -> None:
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")


def preview_dataframe(name: str, df: pd.DataFrame, n: int = 3) -> None:
    print(f"\n--- {name.upper()} ---")
    print(f"Number of samples: {len(df)}")
    if len(df) == 0:
        print("(empty dataframe)")
        return
    print(df.head(n).to_string(index=False))


# =========================
# Text datasets (HF)
# =========================
def prepare_sst2(raw_root: Path) -> None:
    from datasets import load_dataset

    dataset_dir = raw_root / "sst2"
    ensure_dir(dataset_dir)

    print("Loading SST-2 from Hugging Face...")
    ds = load_dataset("glue", "sst2")

    for split_name, split_data in ds.items():
        df = pd.DataFrame(split_data)
        preview_dataframe(f"sst2/{split_name}", df)
        save_dataframe(df, dataset_dir / f"{split_name}.csv")

    print("\nSST-2 raw preparation completed.")


def prepare_ag_news(raw_root: Path) -> None:
    from datasets import load_dataset

    dataset_dir = raw_root / "ag_news"
    ensure_dir(dataset_dir)

    print("Loading AG News from Hugging Face...")
    ds = load_dataset("ag_news")

    for split_name, split_data in ds.items():
        df = pd.DataFrame(split_data)
        preview_dataframe(f"ag_news/{split_name}", df)
        save_dataframe(df, dataset_dir / f"{split_name}.csv")

    print("\nAG News raw preparation completed.")


# =========================
# Image dataset (torchvision)
# =========================
def prepare_cifar10(raw_root: Path) -> None:
    from torchvision.datasets import CIFAR10

    dataset_dir = raw_root / "cifar10"
    ensure_dir(dataset_dir)

    print("Downloading/loading CIFAR-10 from torchvision...")

    train_ds = CIFAR10(root=str(dataset_dir), train=True, download=True)
    test_ds = CIFAR10(root=str(dataset_dir), train=False, download=True)

    class_names = list(train_ds.classes)

    def cifar_to_dataframe(ds: Any) -> pd.DataFrame:
        rows = []
        for idx, (_, label) in enumerate(ds):
            rows.append(
                {
                    "index": idx,
                    "label": int(label),
                    "label_name": class_names[int(label)],
                }
            )
        return pd.DataFrame(rows)

    train_df = cifar_to_dataframe(train_ds)
    test_df = cifar_to_dataframe(test_ds)

    preview_dataframe("cifar10/train", train_df)
    preview_dataframe("cifar10/test", test_df)

    save_dataframe(train_df, dataset_dir / "train_labels.csv")
    save_dataframe(test_df, dataset_dir / "test_labels.csv")

    meta_df = pd.DataFrame(
        {
            "label": list(range(len(class_names))),
            "label_name": class_names,
        }
    )
    save_dataframe(meta_df, dataset_dir / "label_map.csv")

    print("\nCIFAR-10 raw preparation completed.")
    print("Note: Raw image files are stored by torchvision under the same root directory.")


# =========================
# Audio dataset (torchaudio)
# =========================
def prepare_speech_commands(raw_root: Path, version: str = "speech_commands_v0.02") -> None:
    """
    Important:
    We intentionally avoid calling ds[idx], because that triggers waveform loading
    via torchaudio.load(), which in your environment requires torchcodec/FFmpeg and fails.

    Instead, we read dataset metadata from internal file lists only.
    """

    from torchaudio.datasets import SPEECHCOMMANDS

    dataset_dir = raw_root / "speech_commands"
    ensure_dir(dataset_dir)

    print(f"Downloading/loading Speech Commands ({version}) from torchaudio...")

    train_ds = SPEECHCOMMANDS(root=str(dataset_dir), subset="training", download=True)
    val_ds = SPEECHCOMMANDS(root=str(dataset_dir), subset="validation", download=True)
    test_ds = SPEECHCOMMANDS(root=str(dataset_dir), subset="testing", download=True)

    def speechcommands_to_dataframe(ds: Any) -> tuple[pd.DataFrame, list[str]]:
        rows = []
        labels = set()

        # ds._walker contains relative file paths like:
        # speech_commands_v0.02/backward/0a7c2a8d_nohash_0.wav
        for idx, rel_path in enumerate(ds._walker):
            rel_path = Path(rel_path)

            label_name = rel_path.parent.name
            filename = rel_path.name
            stem = rel_path.stem  # e.g. 0a7c2a8d_nohash_0

            labels.add(label_name)

            speaker_id = ""
            utterance_number = -1

            # Expected format: {speaker_id}_nohash_{utterance_number}
            if "_nohash_" in stem:
                speaker_id, utt_str = stem.split("_nohash_", maxsplit=1)
                try:
                    utterance_number = int(utt_str)
                except ValueError:
                    utterance_number = -1
            else:
                speaker_id = stem

            rows.append(
                {
                    "index": idx,
                    "relative_path": str(rel_path).replace("\\", "/"),
                    "filename": filename,
                    "label_name": label_name,
                    "speaker_id": speaker_id,
                    "utterance_number": utterance_number,
                }
            )

        return pd.DataFrame(rows), sorted(labels)

    train_df, train_labels = speechcommands_to_dataframe(train_ds)
    val_df, val_labels = speechcommands_to_dataframe(val_ds)
    test_df, test_labels = speechcommands_to_dataframe(test_ds)

    all_labels = sorted(set(train_labels) | set(val_labels) | set(test_labels))

    preview_dataframe("speech_commands/train", train_df)
    preview_dataframe("speech_commands/validation", val_df)
    preview_dataframe("speech_commands/test", test_df)

    save_dataframe(train_df, dataset_dir / "train_metadata.csv")
    save_dataframe(val_df, dataset_dir / "validation_metadata.csv")
    save_dataframe(test_df, dataset_dir / "test_metadata.csv")

    label_map_df = pd.DataFrame(
        {
            "label_id": list(range(len(all_labels))),
            "label_name": all_labels,
        }
    )
    save_dataframe(label_map_df, dataset_dir / "label_map.csv")

    print("\nSpeech Commands raw preparation completed.")
    print("Note: Raw audio files are stored by torchaudio under the same root directory.")
    print("Note: This metadata-only version avoids waveform decoding and does not require torchcodec.")


# =========================
# Dispatcher
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare raw datasets for modality-agnostic embedding experiments."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sst2", "ag_news", "cifar10", "speech_commands", "all"],
        help="Which dataset to prepare.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    raw_root = project_root / "data" / "raw"
    ensure_dir(raw_root)

    if args.dataset in ("sst2", "all"):
        prepare_sst2(raw_root)

    if args.dataset in ("ag_news", "all"):
        prepare_ag_news(raw_root)

    if args.dataset in ("cifar10", "all"):
        prepare_cifar10(raw_root)

    if args.dataset in ("speech_commands", "all"):
        prepare_speech_commands(raw_root)

    print("\nAll requested raw dataset preparation steps completed successfully.")


if __name__ == "__main__":
    main()
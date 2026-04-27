from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm


# =========================
# General utilities
# =========================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_device(device_arg: Optional[str] = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def infer_column(df: pd.DataFrame, candidates: List[str], required: bool = True) -> Optional[str]:
    lower_map = {col.lower(): col for col in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    if required:
        raise KeyError(
            f"Could not find any of columns {candidates} in metadata file. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def create_train_val_split(
    dataset: Dataset,
    val_ratio: float,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    n_total = len(dataset)
    n_val = int(n_total * val_ratio)
    n_train = n_total - n_val

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=generator)
    return train_ds, val_ds


def cleanup_previous_split_outputs(split_dir: Path) -> None:
    if not split_dir.exists():
        return
    for p in split_dir.glob("*.pt"):
        p.unlink()


def make_output_filename(part_idx: int, single_file: bool) -> str:
    if single_file:
        return "data.pt"
    return f"shard_{part_idx:05d}.pt"


def cat_if_any(tensors: List[torch.Tensor]) -> Optional[torch.Tensor]:
    if len(tensors) == 0:
        return None
    return torch.cat(tensors, dim=0)


# =========================
# Patch Embedding Surgery
# =========================
def build_native_vit(
    backbone: str,
    image_size: int,
    patch_size: int = 4,
    device: torch.device = torch.device("cpu"),
):
    """
    Load a pretrained ViT and surgically replace:
      - patch_embed.proj  (kernel 16x16 -> patch_size x patch_size)
      - pos_embed         (197 tokens -> n_patches+1 tokens)

    All transformer block weights are kept exactly as pretrained.

    The patch projection kernel is re-initialized by spatially average-pooling
    the pretrained 16x16 weights down to patch_size x patch_size, which
    transfers learned color/edge priors into the new smaller kernel.

    Positional embeddings are bicubic-interpolated from the original 14x14
    grid to the new grid size — the same technique used in the ViT paper
    (Section 3.1) and by DeiT/MAE for resolution fine-tuning.

    Returns: (model, image_processor, n_tokens)
      n_tokens = n_patches + 1  (CLS token)
    """
    from transformers import AutoModel, AutoImageProcessor

    model = AutoModel.from_pretrained(backbone)
    processor = AutoImageProcessor.from_pretrained(backbone)

    # ── 1. Read architecture dimensions ───────────────────────────────
    old_proj: nn.Conv2d = model.embeddings.patch_embeddings.projection
    hidden_dim   = old_proj.out_channels   # e.g. 768 for ViT-base
    in_channels  = old_proj.in_channels    # 3 (RGB)
    old_kernel   = old_proj.kernel_size[0] # 16 for standard ViT-base/16

    n_patches = (image_size // patch_size) ** 2   # 49 (MNIST 28x28/4) or 64 (CIFAR 32x32/4)
    n_tokens  = n_patches + 1                     # +1 for CLS

    # ── 2. Build new patch projection ─────────────────────────────────
    new_proj = nn.Conv2d(
        in_channels, hidden_dim,
        kernel_size=patch_size,
        stride=patch_size,
        bias=(old_proj.bias is not None),
    )

    with torch.no_grad():
        old_weight = old_proj.weight.data              # [D, 3, 16, 16]
        # Spatially pool old_kernel x old_kernel -> patch_size x patch_size
        w = old_weight.view(hidden_dim * in_channels, 1, old_kernel, old_kernel)
        w = F.adaptive_avg_pool2d(w, output_size=(patch_size, patch_size))
        new_proj.weight.data = w.view(hidden_dim, in_channels, patch_size, patch_size)

        if old_proj.bias is not None:
            new_proj.bias.data = old_proj.bias.data.clone()

    # ── 3. Swap the projection layer ──────────────────────────────────
    model.embeddings.patch_embeddings.projection = new_proj

    # Update config attributes so any downstream code reads correctly.
    # IMPORTANT: ViTPatchEmbeddings stores image_size as a tuple (H, W),
    # so we must set it as a tuple — setting it as a plain int causes
    # "TypeError: 'int' object is not subscriptable" inside modeling_vit.py.
    model.config.patch_size = patch_size
    model.config.image_size = image_size
    if hasattr(model.embeddings.patch_embeddings, "image_size"):
        model.embeddings.patch_embeddings.image_size = (image_size, image_size)  # must be tuple
    if hasattr(model.embeddings.patch_embeddings, "num_patches"):
        model.embeddings.patch_embeddings.num_patches = n_patches

    # ── 4. Interpolate positional embeddings ──────────────────────────
    # Original: [1, 197, D]  (CLS + 14x14 = 196 patches for 224x224/patch16)
    # Target  : [1, n_tokens, D]
    old_pos_embed = model.embeddings.position_embeddings.data  # [1, 197, D]

    new_pos_embed = nn.Parameter(torch.empty(1, n_tokens, hidden_dim))
    with torch.no_grad():
        # CLS token: keep exactly as pretrained
        new_pos_embed[:, 0, :] = old_pos_embed[:, 0, :]

        # Patch positions: bicubic interpolation from old grid to new grid
        old_grid = int((old_pos_embed.shape[1] - 1) ** 0.5)  # 14
        new_grid = int(n_patches ** 0.5)                       # 7 or 8

        patch_pos = old_pos_embed[:, 1:, :]                   # [1, 196, D]
        patch_pos = patch_pos.reshape(1, old_grid, old_grid, hidden_dim)
        patch_pos = patch_pos.permute(0, 3, 1, 2)             # [1, D, 14, 14]
        patch_pos = F.interpolate(
            patch_pos,
            size=(new_grid, new_grid),
            mode="bicubic",
            align_corners=False,
        )                                                      # [1, D, new_grid, new_grid]
        patch_pos = patch_pos.permute(0, 2, 3, 1)             # [1, new_grid, new_grid, D]
        patch_pos = patch_pos.reshape(1, n_patches, hidden_dim)

        new_pos_embed[:, 1:, :] = patch_pos

    model.embeddings.position_embeddings = new_pos_embed

    model = model.to(device)
    model.eval()

    print(f"\n[PatchSurgery] backbone    : {backbone}")
    print(f"               image_size  : {image_size}x{image_size}")
    print(f"               patch_size  : {patch_size}x{patch_size}")
    print(f"               n_patches   : {n_patches}  ({new_grid}x{new_grid} grid)")
    print(f"               n_tokens    : {n_tokens}  (patches + CLS)")
    print(f"               hidden_dim  : {hidden_dim}")
    print(f"               pos_embed   : {tuple(old_pos_embed.shape)} -> {tuple(new_pos_embed.shape)}")
    print(f"               patch_proj  : kernel {old_kernel}x{old_kernel} -> {patch_size}x{patch_size} (avg-pooled)")

    return model, processor, n_tokens


# =========================
# Streaming shard writer
# =========================
class StreamingShardWriter:
    def __init__(
        self,
        output_dir: Path,
        split_name: str,
        dataset_name: str,
        modality: str,
        backbone: str,
        num_classes: int,
        sequence_type: str,
        max_length: Optional[int],
        max_samples_per_file: int,
    ) -> None:
        self.dataset_output_dir = output_dir
        self.split_name = split_name
        self.split_dir = self.dataset_output_dir / split_name
        ensure_dir(self.split_dir)
        cleanup_previous_split_outputs(self.split_dir)

        self.dataset_name = dataset_name
        self.modality = modality
        self.backbone = backbone
        self.num_classes = num_classes
        self.sequence_type = sequence_type
        self.max_length = max_length
        self.max_samples_per_file = max_samples_per_file

        if self.max_samples_per_file <= 0:
            raise ValueError("--max-samples-per-file must be > 0")

        self._embeddings: List[torch.Tensor] = []
        self._attention_masks: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []

        self._sentences: List[str] = []
        self._paths: List[str] = []
        self._label_names: List[str] = []
        self._speaker_ids: List[Optional[str]] = []
        self._utterance_numbers: List[Optional[int]] = []
        self._sample_rates: List[int] = []

        self.current_samples = 0
        self.part_idx = 0
        self.num_parts_written = 0
        self.total_samples_seen = 0

    def add_batch(self, batch_data: Dict[str, Any]) -> None:
        batch_size = int(batch_data["embeddings"].shape[0])
        start = 0

        while start < batch_size:
            remaining_space = self.max_samples_per_file - self.current_samples
            end = min(start + remaining_space, batch_size)

            chunk_embeddings = batch_data["embeddings"][start:end]
            chunk_attention_mask = batch_data["attention_mask"][start:end]
            chunk_size = int(chunk_embeddings.shape[0])

            self._embeddings.append(chunk_embeddings)
            self._attention_masks.append(chunk_attention_mask)

            self.current_samples += chunk_size
            self.total_samples_seen += chunk_size

            if "labels" in batch_data and batch_data["labels"] is not None:
                self._labels.append(batch_data["labels"][start:end])

            if "sentences" in batch_data and batch_data["sentences"] is not None:
                self._sentences.extend(batch_data["sentences"][start:end])

            if "paths" in batch_data and batch_data["paths"] is not None:
                self._paths.extend(batch_data["paths"][start:end])

            if "label_names" in batch_data and batch_data["label_names"] is not None:
                self._label_names.extend(batch_data["label_names"][start:end])

            if "speaker_ids" in batch_data and batch_data["speaker_ids"] is not None:
                self._speaker_ids.extend(batch_data["speaker_ids"][start:end])

            if "utterance_numbers" in batch_data and batch_data["utterance_numbers"] is not None:
                self._utterance_numbers.extend(batch_data["utterance_numbers"][start:end])

            if "sample_rates" in batch_data and batch_data["sample_rates"] is not None:
                self._sample_rates.extend(batch_data["sample_rates"][start:end])

            start = end

            if self.current_samples == self.max_samples_per_file:
                self.flush(force_shard=True, final_flush=False)

    def flush(self, force_shard: bool = False, final_flush: bool = False) -> None:
        if self.current_samples == 0:
            return

        embeddings = cat_if_any(self._embeddings)
        attention_mask = cat_if_any(self._attention_masks)
        labels = cat_if_any(self._labels)

        if embeddings is None or attention_mask is None:
            raise RuntimeError("Buffer corrupted: embeddings or attention_mask missing.")

        single_file = final_flush and self.num_parts_written == 0 and not force_shard
        filename = make_output_filename(self.part_idx, single_file=single_file)

        save_obj: Dict[str, Any] = {
            "embeddings": embeddings,
            "attention_mask": attention_mask,
            "meta": {
                "dataset_name": self.dataset_name,
                "modality": self.modality,
                "backbone": self.backbone,
                "num_classes": self.num_classes,
                "split": self.split_name,
                "sequence_type": self.sequence_type,
                "max_length": self.max_length,
                "part_idx": self.part_idx,
                "num_parts": 1 if single_file else None,
                "num_samples": int(embeddings.shape[0]),
                "storage_mode": "single_file" if single_file else "sharded",
            },
        }

        if labels is not None:
            save_obj["labels"] = labels
        if len(self._sentences) > 0:
            save_obj["sentences"] = list(self._sentences)
        if len(self._paths) > 0:
            save_obj["paths"] = list(self._paths)
        if len(self._label_names) > 0:
            save_obj["label_names"] = list(self._label_names)
        if len(self._speaker_ids) > 0:
            save_obj["speaker_ids"] = list(self._speaker_ids)
        if len(self._utterance_numbers) > 0:
            save_obj["utterance_numbers"] = list(self._utterance_numbers)
        if len(self._sample_rates) > 0:
            save_obj["sample_rates"] = list(self._sample_rates)

        output_path = self.split_dir / filename
        torch.save(save_obj, output_path)

        print(f"Saved {self.dataset_name}/{self.split_name} -> {output_path}")
        print(f"  embeddings shape     : {tuple(save_obj['embeddings'].shape)}")
        print(f"  attention_mask shape : {tuple(save_obj['attention_mask'].shape)}")
        if "labels" in save_obj:
            print(f"  labels shape         : {tuple(save_obj['labels'].shape)}")
        else:
            print("  labels               : not available")
        if "sentences" in save_obj:
            print(f"  number of sentences  : {len(save_obj['sentences'])}")
        if "paths" in save_obj:
            print(f"  number of paths      : {len(save_obj['paths'])}")

        self.num_parts_written += 1
        self.part_idx += 1

        self._embeddings.clear()
        self._attention_masks.clear()
        self._labels.clear()
        self._sentences.clear()
        self._paths.clear()
        self._label_names.clear()
        self._speaker_ids.clear()
        self._utterance_numbers.clear()
        self._sample_rates.clear()
        self.current_samples = 0

    def finalize(self) -> None:
        if self.current_samples == 0 and self.num_parts_written == 0:
            raise RuntimeError(
                f"No samples were written for split '{self.split_name}'. Check dataset/loader."
            )

        if self.num_parts_written == 0:
            self.flush(force_shard=False, final_flush=True)
        else:
            self.flush(force_shard=True, final_flush=True)

        print(
            f"Finished {self.dataset_name}/{self.split_name} | "
            f"total_samples={self.total_samples_seen} | files_written={self.num_parts_written}"
        )


# =========================
# Text datasets
# =========================
class TextCSVDataset(Dataset):
    def __init__(self, csv_path: Path, text_col: str, label_col: Optional[str] = None):
        self.df = pd.read_csv(csv_path)

        if text_col not in self.df.columns:
            raise KeyError(f"Missing text column '{text_col}' in {csv_path}")

        self.text_col = text_col
        self.label_col = label_col if (label_col is not None and label_col in self.df.columns) else None

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        item = {"text": str(row[self.text_col])}

        if self.label_col is not None and pd.notna(row[self.label_col]):
            item["label"] = int(row[self.label_col])

        return item


def collate_text(batch: List[Dict[str, Any]], tokenizer, max_length: int) -> Dict[str, Any]:
    texts = [item["text"] for item in batch]

    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    result = {
        "texts": texts,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }

    if "label" in batch[0]:
        result["labels"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    return result


@torch.no_grad()
def extract_text_embeddings_to_files(
    split_name: str,
    dataset: Dataset,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    num_workers: int,
    keep_text: bool,
    output_dir: Path,
    dataset_name: str,
    backbone: str,
    num_classes: int,
    max_samples_per_file: int,
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_text(batch, tokenizer, max_length),
    )

    writer = StreamingShardWriter(
        output_dir=output_dir,
        split_name=split_name,
        dataset_name=dataset_name,
        modality="text",
        backbone=backbone,
        num_classes=num_classes,
        sequence_type="tokens",
        max_length=max_length,
        max_samples_per_file=max_samples_per_file,
    )

    model.eval()

    for batch in tqdm(loader, desc=f"Extracting text embeddings [{split_name}]"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state.detach().cpu()

        batch_data: Dict[str, Any] = {
            "embeddings": last_hidden_state,
            "attention_mask": batch["attention_mask"].cpu(),
        }

        if "labels" in batch:
            batch_data["labels"] = batch["labels"].cpu()

        if keep_text:
            batch_data["sentences"] = batch["texts"]

        writer.add_batch(batch_data)

    writer.finalize()


# =========================
# Image datasets
# =========================
class CIFAR10Wrapper(Dataset):
    def __init__(self, root: Path, train: bool):
        from torchvision.datasets import CIFAR10
        self.ds = CIFAR10(root=str(root), train=train, download=False)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, label = self.ds[idx]
        return {"image": image, "label": int(label)}


class MNISTWrapper(Dataset):
    def __init__(self, root: Path, train: bool):
        from torchvision.datasets import MNIST
        from torchvision import transforms

        self.ds = MNIST(
            root=str(root),
            train=train,
            download=False,
            transform=transforms.Lambda(lambda img: img.convert("RGB")),
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        image, label = self.ds[idx]
        return {"image": image, "label": int(label)}


def collate_image(
    batch: List[Dict[str, Any]],
    image_processor,
    image_size: int,
) -> Dict[str, Any]:
    """
    Collate images at their NATIVE resolution (image_size x image_size).

    We explicitly override the processor's default resize target so images
    are NOT upsampled to 224x224. This is what allows the surgically-patched
    ViT to produce compact token sequences (49 for MNIST, 64 for CIFAR10).
    """
    images = [item["image"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)

    processed = image_processor(
        images=images,
        return_tensors="pt",
        do_resize=True,
        size={"height": image_size, "width": image_size},  # Force native resolution
        do_rescale=True,
        do_normalize=True,
    )

    return {
        "pixel_values": processed["pixel_values"],
        "labels": labels,
    }


@torch.no_grad()
def extract_image_embeddings_to_files(
    split_name: str,
    dataset: Dataset,
    image_processor,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    output_dir: Path,
    dataset_name: str,
    backbone: str,
    num_classes: int,
    max_samples_per_file: int,
    image_size: int = 224,   # passed through from prepare_* so collate uses native size
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_image(batch, image_processor, image_size=image_size),
    )

    writer = StreamingShardWriter(
        output_dir=output_dir,
        split_name=split_name,
        dataset_name=dataset_name,
        modality="image",
        backbone=backbone,
        num_classes=num_classes,
        sequence_type="patches",
        max_length=None,
        max_samples_per_file=max_samples_per_file,
    )

    model.eval()

    for batch in tqdm(loader, desc=f"Extracting image embeddings [{split_name}]"):
        pixel_values = batch["pixel_values"].to(device)
        outputs = model(pixel_values=pixel_values)

        if not hasattr(outputs, "last_hidden_state"):
            raise ValueError("Image backbone does not expose last_hidden_state.")

        hidden = outputs.last_hidden_state.detach().cpu()
        B, T, _ = hidden.shape
        attention_mask = torch.ones(B, T, dtype=torch.long)

        batch_data = {
            "embeddings": hidden,
            "attention_mask": attention_mask,
            "labels": batch["labels"].cpu(),
        }

        writer.add_batch(batch_data)

    writer.finalize()


# =========================
# Audio dataset
# =========================
class SpeechCommandsMetadataDataset(Dataset):
    def __init__(self, root: Path, metadata_csv: Path):
        self.root = root
        self.metadata_csv = metadata_csv
        self.df = pd.read_csv(metadata_csv)

        if len(self.df) == 0:
            raise ValueError(f"Metadata CSV is empty: {metadata_csv}")

        self.audio_root = self._find_audio_root(root)

        self.path_col = infer_column(
            self.df,
            candidates=[
                "path", "filepath", "file_path", "file", "filename",
                "file_name", "relative_path", "relpath", "audio_path", "audio"
            ],
            required=False,
        )

        self.label_col = infer_column(
            self.df,
            candidates=["label", "label_name", "command", "keyword", "class"],
            required=False,
        )

        self.speaker_col = infer_column(
            self.df,
            candidates=["speaker_id", "speaker", "speakerid"],
            required=False,
        )

        self.utt_col = infer_column(
            self.df,
            candidates=["utterance_number", "utterance_id", "utterance", "index"],
            required=False,
        )

        self.sample_rate_col = infer_column(
            self.df,
            candidates=["sample_rate", "sr"],
            required=False,
        )

        if self.path_col is None:
            self._build_recursive_index()

        self.label_to_id = self._build_label_map()

    def _find_audio_root(self, root: Path) -> Path:
        candidates = [
            root / "SpeechCommands" / "speech_commands_v0.02",
            root / "SpeechCommands",
            root / "speech_commands_v0.02",
            root,
        ]
        for c in candidates:
            if c.exists():
                wavs = list(c.rglob("*.wav"))
                if len(wavs) > 0:
                    return c
        raise FileNotFoundError(f"Could not find audio root with .wav files under {root}")

    def _build_recursive_index(self) -> None:
        self.basename_to_paths: Dict[str, List[Path]] = {}
        for wav_path in self.audio_root.rglob("*.wav"):
            self.basename_to_paths.setdefault(wav_path.name, []).append(wav_path)

    def _build_label_map(self) -> Dict[str, int]:
        if self.label_col is not None:
            labels = sorted(self.df[self.label_col].astype(str).unique().tolist())
        else:
            labels = sorted({p.parent.name for p in self.audio_root.rglob("*.wav")})
        return {label: idx for idx, label in enumerate(labels)}

    def _resolve_audio_path(self, row: pd.Series) -> Path:
        if self.path_col is not None:
            raw_path = str(row[self.path_col])

            candidate_paths = [
                Path(raw_path),
                self.root / raw_path,
                self.audio_root / raw_path,
            ]

            if self.label_col is not None:
                label_name = str(row[self.label_col])
                candidate_paths.append(self.audio_root / label_name / Path(raw_path).name)

            for p in candidate_paths:
                if p.exists():
                    return p.resolve()

            raise FileNotFoundError(
                f"Could not resolve audio path '{raw_path}' from metadata file {self.metadata_csv}"
            )

        if self.label_col is None:
            raise KeyError(
                "Metadata CSV has no path-like column and no label column; cannot resolve audio file."
            )

        label_name = str(row[self.label_col])

        possible_name_cols = ["file_name", "filename", "file", "audio", "path"]
        existing_name_cols = [c for c in possible_name_cols if c in self.df.columns]

        for col in existing_name_cols:
            filename = str(row[col])
            p = self.audio_root / label_name / filename
            if p.exists():
                return p.resolve()

        for col in existing_name_cols:
            filename = str(row[col])
            if filename in self.basename_to_paths:
                matches = self.basename_to_paths[filename]
                if len(matches) == 1:
                    return matches[0].resolve()
                for m in matches:
                    if m.parent.name == label_name:
                        return m.resolve()

        raise FileNotFoundError(
            f"Could not infer audio file path for row from {self.metadata_csv}"
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        audio_path = self._resolve_audio_path(row)

        waveform, sample_rate = torchaudio.load(str(audio_path))

        if self.label_col is not None:
            label_name = str(row[self.label_col])
            label_id = self.label_to_id[label_name]
        else:
            label_name = audio_path.parent.name
            label_id = self.label_to_id[label_name]

        speaker_id = None
        if self.speaker_col is not None and pd.notna(row[self.speaker_col]):
            speaker_id = str(row[self.speaker_col])

        utterance_number = None
        if self.utt_col is not None and pd.notna(row[self.utt_col]):
            try:
                utterance_number = int(row[self.utt_col])
            except Exception:
                utterance_number = None

        if self.sample_rate_col is not None and pd.notna(row[self.sample_rate_col]):
            try:
                sample_rate = int(row[self.sample_rate_col])
            except Exception:
                pass

        return {
            "waveform": waveform,
            "sample_rate": int(sample_rate),
            "label": int(label_id),
            "label_name": label_name,
            "speaker_id": speaker_id,
            "utterance_number": utterance_number,
            "path": str(audio_path),
        }


def collate_audio(
    batch: List[Dict[str, Any]],
    processor,
    max_audio_length: Optional[int] = None,
    target_sample_rate: int = 16000,
) -> Dict[str, Any]:
    waveforms = []
    labels = []
    paths = []
    label_names = []
    speaker_ids = []
    utterance_numbers = []
    sample_rates = []

    for item in batch:
        waveform = item["waveform"]
        sr = int(item["sample_rate"])

        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform = waveform.squeeze(0)

        if sr != target_sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, target_sample_rate)
            sr = target_sample_rate

        if max_audio_length is not None:
            waveform = waveform[:max_audio_length]

        waveforms.append(waveform.numpy())
        labels.append(item["label"])
        paths.append(item["path"])
        label_names.append(item["label_name"])
        speaker_ids.append(item["speaker_id"])
        utterance_numbers.append(item["utterance_number"])
        sample_rates.append(sr)

    processed = processor(
        waveforms,
        sampling_rate=target_sample_rate,
        return_tensors="pt",
        padding=True,
    )

    return {
        "input_values": processed["input_values"],
        "attention_mask": processed.get("attention_mask", None),
        "labels": torch.tensor(labels, dtype=torch.long),
        "paths": paths,
        "label_names": label_names,
        "speaker_ids": speaker_ids,
        "utterance_numbers": utterance_numbers,
        "sample_rates": sample_rates,
    }


@torch.no_grad()
def extract_audio_embeddings_to_files(
    split_name: str,
    dataset: Dataset,
    processor,
    model,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_audio_length: Optional[int],
    output_dir: Path,
    dataset_name: str,
    backbone: str,
    num_classes: int,
    max_samples_per_file: int,
) -> None:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_audio(batch, processor, max_audio_length),
    )

    writer = StreamingShardWriter(
        output_dir=output_dir,
        split_name=split_name,
        dataset_name=dataset_name,
        modality="audio",
        backbone=backbone,
        num_classes=num_classes,
        sequence_type="frames",
        max_length=max_audio_length,
        max_samples_per_file=max_samples_per_file,
    )

    model.eval()

    for batch in tqdm(loader, desc=f"Extracting audio embeddings [{split_name}]"):
        input_values = batch["input_values"].to(device)
        kwargs = {"input_values": input_values}

        if batch["attention_mask"] is not None:
            kwargs["attention_mask"] = batch["attention_mask"].to(device)

        outputs = model(**kwargs)
        hidden = outputs.last_hidden_state.detach().cpu()

        B, T, _ = hidden.shape

        if batch["attention_mask"] is not None:
            attn = batch["attention_mask"].cpu()
        else:
            attn = torch.ones(B, T, dtype=torch.long)

        if attn.shape[1] != T:
            attn = torch.ones(B, T, dtype=torch.long)

        batch_data: Dict[str, Any] = {
            "embeddings": hidden,
            "attention_mask": attn,
            "labels": batch["labels"].cpu(),
            "paths": batch["paths"],
            "label_names": batch["label_names"],
            "speaker_ids": batch["speaker_ids"],
            "utterance_numbers": batch["utterance_numbers"],
            "sample_rates": batch["sample_rates"],
        }

        writer.add_batch(batch_data)

    writer.finalize()


# =========================
# Dataset-specific runners
# =========================
def prepare_sst2(
    project_root: Path,
    device: torch.device,
    backbone: str,
    max_length: int,
    batch_size: int,
    num_workers: int,
    keep_text: bool,
    max_samples_per_file: int,
) -> None:
    from transformers import AutoModel, AutoTokenizer

    raw_dir = project_root / "data" / "raw" / "sst2"
    out_dir = project_root / "data" / "processed" / "sst2"
    ensure_dir(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone).to(device)

    splits = {
        "train": TextCSVDataset(raw_dir / "train.csv", text_col="sentence", label_col="label"),
        "validation": TextCSVDataset(raw_dir / "validation.csv", text_col="sentence", label_col="label"),
        "test": TextCSVDataset(raw_dir / "test.csv", text_col="sentence", label_col="label"),
    }

    for split_name, split_ds in splits.items():
        print(f"\nProcessing sst2/{split_name}")
        extract_text_embeddings_to_files(
            split_name=split_name,
            dataset=split_ds,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers,
            keep_text=keep_text,
            output_dir=out_dir,
            dataset_name="sst2",
            backbone=backbone,
            num_classes=2,
            max_samples_per_file=max_samples_per_file,
        )


def prepare_ag_news(
    project_root: Path,
    device: torch.device,
    backbone: str,
    max_length: int,
    batch_size: int,
    num_workers: int,
    keep_text: bool,
    val_ratio: float,
    seed: int,
    max_samples_per_file: int,
) -> None:
    from transformers import AutoModel, AutoTokenizer

    raw_dir = project_root / "data" / "raw" / "ag_news"
    out_dir = project_root / "data" / "processed" / "ag_news"
    ensure_dir(out_dir)

    tokenizer = AutoTokenizer.from_pretrained(backbone)
    model = AutoModel.from_pretrained(backbone).to(device)

    train_full = TextCSVDataset(raw_dir / "train.csv", text_col="text", label_col="label")
    test_ds = TextCSVDataset(raw_dir / "test.csv", text_col="text", label_col="label")
    train_ds, val_ds = create_train_val_split(train_full, val_ratio=val_ratio, seed=seed)

    splits = {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }

    for split_name, split_ds in splits.items():
        print(f"\nProcessing ag_news/{split_name}")
        extract_text_embeddings_to_files(
            split_name=split_name,
            dataset=split_ds,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=batch_size,
            max_length=max_length,
            num_workers=num_workers,
            keep_text=keep_text,
            output_dir=out_dir,
            dataset_name="ag_news",
            backbone=backbone,
            num_classes=4,
            max_samples_per_file=max_samples_per_file,
        )


def prepare_cifar10(
    project_root: Path,
    device: torch.device,
    backbone: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    max_samples_per_file: int,
    patch_size: int = 4,
) -> None:
    """
    CIFAR-10: 32x32 images -> patch_size=4 -> 8x8 grid = 64 patches + 1 CLS = 65 tokens.
    Uses patch embedding surgery on the pretrained ViT backbone.
    """
    raw_dir = project_root / "data" / "raw" / "cifar10"
    out_dir = project_root / "data" / "processed" / "cifar10"
    ensure_dir(out_dir)

    image_size = 32
    model, image_processor, n_tokens = build_native_vit(
        backbone=backbone,
        image_size=image_size,
        patch_size=patch_size,
        device=device,
    )
    print(f"CIFAR10: {n_tokens} tokens per image  (64 patches + 1 CLS)")

    train_full = CIFAR10Wrapper(root=raw_dir, train=True)
    test_ds    = CIFAR10Wrapper(root=raw_dir, train=False)
    train_ds, val_ds = create_train_val_split(train_full, val_ratio=val_ratio, seed=seed)

    splits = {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }

    for split_name, split_ds in splits.items():
        print(f"\nProcessing cifar10/{split_name}")
        extract_image_embeddings_to_files(
            split_name=split_name,
            dataset=split_ds,
            image_processor=image_processor,
            model=model,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            output_dir=out_dir,
            dataset_name="cifar10",
            backbone=backbone,
            num_classes=10,
            max_samples_per_file=max_samples_per_file,
            image_size=image_size,
        )


def prepare_mnist(
    project_root: Path,
    device: torch.device,
    backbone: str,
    batch_size: int,
    num_workers: int,
    val_ratio: float,
    seed: int,
    max_samples_per_file: int,
    patch_size: int = 4,
) -> None:
    """
    MNIST: 28x28 images -> patch_size=4 -> 7x7 grid = 49 patches + 1 CLS = 50 tokens.
    Uses patch embedding surgery on the pretrained ViT backbone.
    """
    raw_dir = project_root / "data" / "raw" / "mnist"
    out_dir = project_root / "data" / "processed" / "mnist"
    ensure_dir(out_dir)

    image_size = 28
    model, image_processor, n_tokens = build_native_vit(
        backbone=backbone,
        image_size=image_size,
        patch_size=patch_size,
        device=device,
    )
    print(f"MNIST: {n_tokens} tokens per image  (49 patches + 1 CLS)")

    train_full = MNISTWrapper(root=raw_dir, train=True)
    test_ds    = MNISTWrapper(root=raw_dir, train=False)
    train_ds, val_ds = create_train_val_split(train_full, val_ratio=val_ratio, seed=seed)

    splits = {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }

    for split_name, split_ds in splits.items():
        print(f"\nProcessing mnist/{split_name}")
        extract_image_embeddings_to_files(
            split_name=split_name,
            dataset=split_ds,
            image_processor=image_processor,
            model=model,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            output_dir=out_dir,
            dataset_name="mnist",
            backbone=backbone,
            num_classes=10,
            max_samples_per_file=max_samples_per_file,
            image_size=image_size,
        )


def prepare_speech_commands(
    project_root: Path,
    device: torch.device,
    backbone: str,
    batch_size: int,
    num_workers: int,
    max_audio_length: Optional[int],
    max_samples_per_file: int,
) -> None:
    from transformers import Wav2Vec2Model, Wav2Vec2Processor

    raw_dir = project_root / "data" / "raw" / "speech_commands"
    out_dir = project_root / "data" / "processed" / "speech_commands"
    ensure_dir(out_dir)

    processor = Wav2Vec2Processor.from_pretrained(backbone)
    model = Wav2Vec2Model.from_pretrained(backbone).to(device)

    splits = {
        "train": SpeechCommandsMetadataDataset(raw_dir, raw_dir / "train_metadata.csv"),
        "validation": SpeechCommandsMetadataDataset(raw_dir, raw_dir / "validation_metadata.csv"),
        "test": SpeechCommandsMetadataDataset(raw_dir, raw_dir / "test_metadata.csv"),
    }

    for split_name, split_ds in splits.items():
        print(f"\nProcessing speech_commands/{split_name}")
        extract_audio_embeddings_to_files(
            split_name=split_name,
            dataset=split_ds,
            processor=processor,
            model=model,
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            max_audio_length=max_audio_length,
            output_dir=out_dir,
            dataset_name="speech_commands",
            backbone=backbone,
            num_classes=len(split_ds.label_to_id),
            max_samples_per_file=max_samples_per_file,
        )


# =========================
# Main
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare modality-agnostic embedding datasets with streaming shard saving."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["sst2", "ag_news", "cifar10", "mnist", "speech_commands", "all"],
    )

    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)

    parser.add_argument("--text-backbone", type=str, default="bert-base-uncased")
    parser.add_argument("--image-backbone", type=str, default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--audio-backbone", type=str, default="facebook/wav2vec2-base-960h")

    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--max-audio-length", type=int, default=None)
    parser.add_argument("--keep-text", action="store_true")

    parser.add_argument(
        "--patch-size",
        type=int,
        default=4,
        help=(
            "Patch size for image ViT surgery. "
            "MNIST 28x28/4 -> 49 tokens. CIFAR10 32x32/4 -> 64 tokens. "
            "Pretrained weights are transferred via kernel avg-pooling + pos-embed interpolation."
        ),
    )

    parser.add_argument(
        "--max-samples-per-file",
        type=int,
        default=1000,
        help="Maximum number of samples in each saved .pt file. If exceeded, shards are written automatically.",
    )

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    project_root = Path(__file__).resolve().parents[1]
    device = get_device(args.device)

    print(f"Using device: {device}")

    if args.dataset in ("sst2", "all"):
        prepare_sst2(
            project_root=project_root,
            device=device,
            backbone=args.text_backbone,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            keep_text=args.keep_text,
            max_samples_per_file=args.max_samples_per_file,
        )

    if args.dataset in ("ag_news", "all"):
        prepare_ag_news(
            project_root=project_root,
            device=device,
            backbone=args.text_backbone,
            max_length=args.max_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            keep_text=args.keep_text,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_samples_per_file=args.max_samples_per_file,
        )

    if args.dataset in ("cifar10", "all"):
        prepare_cifar10(
            project_root=project_root,
            device=device,
            backbone=args.image_backbone,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_samples_per_file=args.max_samples_per_file,
            patch_size=args.patch_size,
        )

    if args.dataset in ("mnist", "all"):
        prepare_mnist(
            project_root=project_root,
            device=device,
            backbone=args.image_backbone,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_ratio=args.val_ratio,
            seed=args.seed,
            max_samples_per_file=args.max_samples_per_file,
            patch_size=args.patch_size,
        )

    if args.dataset in ("speech_commands", "all"):
        prepare_speech_commands(
            project_root=project_root,
            device=device,
            backbone=args.audio_backbone,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_audio_length=args.max_audio_length,
            max_samples_per_file=args.max_samples_per_file,
        )

    print("\nEmbedding preparation completed successfully.")


if __name__ == "__main__":
    main()

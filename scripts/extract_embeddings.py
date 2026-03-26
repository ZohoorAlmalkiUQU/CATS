import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def get_device(device_arg: Optional[str] = None) -> torch.device:
    if device_arg is not None:
        return torch.device(device_arg)

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def collate_batch(batch, tokenizer, max_length: int):
    sentences = [item["sentence"] for item in batch]

    tokenized = tokenizer(
        sentences,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    result = {
        "sentences": sentences,
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
    }

    # SST-2 test split usually has no labels
    if "label" in batch[0]:
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        result["labels"] = labels

    return result


@torch.no_grad()
def extract_split_embeddings(
    split_dataset,
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
    num_workers: int,
    keep_sentences: bool,
) -> Dict[str, Any]:
    loader = DataLoader(
        split_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: collate_batch(batch, tokenizer, max_length),
    )

    all_embeddings = []
    all_attention_masks = []
    all_labels = []
    all_sentences = []

    model.eval()

    for batch in tqdm(loader, desc="Extracting"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        # Last hidden state: [B, T, D]
        last_hidden_state = outputs.last_hidden_state.detach().cpu()

        all_embeddings.append(last_hidden_state)
        all_attention_masks.append(batch["attention_mask"].cpu())

        if "labels" in batch:
            all_labels.append(batch["labels"].cpu())

        if keep_sentences:
            all_sentences.extend(batch["sentences"])

    result = {
        "embeddings": torch.cat(all_embeddings, dim=0),         # [N, T, D]
        "attention_mask": torch.cat(all_attention_masks, dim=0) # [N, T]
    }

    if all_labels:
        result["labels"] = torch.cat(all_labels, dim=0)         # [N]

    if keep_sentences:
        result["sentences"] = all_sentences

    return result


def save_split(
    split_name: str,
    split_data: Dict[str, Any],
    output_dir: Path,
    model_name: str,
    max_length: int,
) -> None:
    save_obj = {
        "split": split_name,
        "model_name": model_name,
        "max_length": max_length,
        **split_data,
    }

    output_path = output_dir / f"{split_name}.pt"
    torch.save(save_obj, output_path)
    print(f"Saved {split_name} -> {output_path}")

    embeddings = save_obj["embeddings"]
    attention_mask = save_obj["attention_mask"]

    print(f"  embeddings shape     : {tuple(embeddings.shape)}")
    print(f"  attention_mask shape : {tuple(attention_mask.shape)}")

    if "labels" in save_obj:
        print(f"  labels shape         : {tuple(save_obj['labels'].shape)}")
    else:
        print("  labels               : not available")

    if "sentences" in save_obj:
        print(f"  number of sentences  : {len(save_obj['sentences'])}")


def main():
    parser = argparse.ArgumentParser(description="Extract SST-2 embeddings using a pretrained Transformer.")
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="e.g. cuda, cpu")
    parser.add_argument("--keep-sentences", action="store_true", help="Save raw sentences for debugging.")
    args, _ = parser.parse_known_args()

    project_root = Path(__file__).resolve().parents[1]
    output_dir = project_root / "data" / "processed" / "sst2"
    ensure_dir(output_dir)

    device = get_device(args.device)
    print(f"Using device: {device}")

    print("Loading SST-2 dataset...")
    dataset = load_dataset("glue", "sst2")

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(f"Loading model: {args.model_name}")
    model = AutoModel.from_pretrained(args.model_name)
    model.to(device)

    for split_name in dataset.keys():
        print(f"\nProcessing split: {split_name}")
        split_dataset = dataset[split_name]

        split_data = extract_split_embeddings(
            split_dataset=split_dataset,
            tokenizer=tokenizer,
            model=model,
            device=device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            num_workers=args.num_workers,
            keep_sentences=args.keep_sentences,
        )

        save_split(
            split_name=split_name,
            split_data=split_data,
            output_dir=output_dir,
            model_name=args.model_name,
            max_length=args.max_length,
        )

    print("\nEmbedding extraction completed successfully.")


if __name__ == "__main__":
    main()

    # From project root: Bash
    # python scripts/extract_embeddings.py --model-name bert-base-uncased --max-length 64 --batch-size 32 --keep-sentences
    # If you have GPU: Bash
    # python scripts/extract_embeddings.py --model-name bert-base-uncased --max-length 64 --batch-size 64 --device cuda --keep-sentences
"""
Training Script
===============
End-to-end training pipeline for the Neural Code De-obfuscator.

Generates obfuscated/clean pairs on-the-fly from a directory of clean
Python files, then trains the Transformer encoder-decoder with
graph spectral feature injection.

Usage:
    python -m src.train --data-dir benchmarks/data/clean --epochs 50
    python -m src.train --data-dir benchmarks/data/clean --epochs 10 --batch-size 8 --lr 3e-4
"""

import argparse
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from src.model import DeobfuscatorModel
from src.parser import parse_source
from src.features import GraphFeatureExtractor
from benchmarks.obfuscate import Obfuscator


# ──────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────

class DeobfuscationDataset(Dataset):
    """
    On-the-fly obfuscation dataset.

    Loads clean Python files, obfuscates them at random levels (1-3),
    and yields (obfuscated_tokens, clean_tokens, graph_features) triples.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing clean .py files.
    max_seq_len : int
        Maximum sequence length (byte-level tokens).
    graph_feature_dim : int
        Expected dimension of the graph feature vector.
    """

    def __init__(
        self,
        data_dir: str | Path,
        max_seq_len: int = 1024,
        graph_feature_dim: int = 130,
    ):
        self.data_dir = Path(data_dir)
        self.max_seq_len = max_seq_len
        self.graph_feature_dim = graph_feature_dim
        self.extractor = GraphFeatureExtractor(num_eigenvalues=32)

        # Load all clean source files
        self.sources = []
        for f in sorted(self.data_dir.glob("*.py")):
            text = f.read_text()
            if text.strip():
                self.sources.append(text)

        if not self.sources:
            raise FileNotFoundError(
                f"No .py files found in {self.data_dir}. "
                f"Run `python benchmarks/fetch_dataset.py` first."
            )

        # Pre-create obfuscators for each level (with different seeds)
        self.obfuscators = {
            level: Obfuscator(level=level, seed=42 + level)
            for level in (1, 2, 3)
        }

    def __len__(self) -> int:
        # Each clean file generates 3 training pairs (one per level)
        return len(self.sources) * 3

    def __getitem__(self, idx: int):
        source_idx = idx // 3
        level = (idx % 3) + 1

        clean = self.sources[source_idx]
        obfuscator = self.obfuscators[level]

        try:
            obfuscated = obfuscator.obfuscate(clean)
        except Exception:
            obfuscated = clean  # Fallback: treat clean as its own input

        # Tokenize to byte-level
        src_tokens = self._tokenize(obfuscated)
        tgt_tokens = self._tokenize(clean)

        # Extract graph features from obfuscated code
        try:
            graph = parse_source(obfuscated)
            features = self.extractor.extract(graph)
            graph_vec = features.feature_vector
        except Exception:
            graph_vec = torch.zeros(self.graph_feature_dim)

        # Ensure graph feature vector is the right size
        if len(graph_vec) != self.graph_feature_dim:
            padded = torch.zeros(self.graph_feature_dim)
            copy_len = min(len(graph_vec), self.graph_feature_dim)
            padded[:copy_len] = graph_vec[:copy_len]
            graph_vec = padded

        return src_tokens, tgt_tokens, graph_vec

    def _tokenize(self, source: str) -> torch.Tensor:
        """Convert source string to byte-level token IDs, truncated to max_seq_len."""
        raw_bytes = source.encode("utf-8", errors="replace")
        tokens = list(raw_bytes[: self.max_seq_len])
        return torch.tensor(tokens, dtype=torch.long)


def collate_fn(batch):
    """
    Pad variable-length sequences to the max length in the batch.

    Returns
    -------
    src_tokens : (batch, max_src_len)
    tgt_tokens : (batch, max_tgt_len)
    graph_features : (batch, graph_feature_dim)
    """
    src_list, tgt_list, feat_list = zip(*batch)

    # Pad source tokens (pad with 0 = null byte)
    max_src = max(len(s) for s in src_list)
    src_padded = torch.zeros(len(src_list), max_src, dtype=torch.long)
    for i, s in enumerate(src_list):
        src_padded[i, : len(s)] = s

    # Pad target tokens
    max_tgt = max(len(t) for t in tgt_list)
    tgt_padded = torch.zeros(len(tgt_list), max_tgt, dtype=torch.long)
    for i, t in enumerate(tgt_list):
        tgt_padded[i, : len(t)] = t

    # Stack graph features
    graph_features = torch.stack(feat_list)

    return src_padded, tgt_padded, graph_features


# ──────────────────────────────────────────────────────────────
# Training Loop
# ──────────────────────────────────────────────────────────────

def train(
    model: DeobfuscatorModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
) -> dict:
    """Run one training epoch. Returns loss and accuracy metrics."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for batch_idx, (src, tgt, graph_feat) in enumerate(dataloader):
        src = src.to(device)
        tgt = tgt.to(device)
        graph_feat = graph_feat.to(device)

        # Teacher forcing: input = tgt[:-1], target = tgt[1:]
        tgt_input = tgt[:, :-1]
        tgt_target = tgt[:, 1:]

        if tgt_input.shape[1] == 0:
            continue

        # Forward pass
        logits = model(src, tgt_input, graph_features=graph_feat)

        # Cross-entropy loss (ignore padding token 0)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            tgt_target.reshape(-1),
            ignore_index=0,
        )

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Track metrics
        total_loss += loss.item()

        # Token accuracy (excluding padding)
        preds = logits.argmax(dim=-1)
        mask = tgt_target != 0
        total_correct += (preds[mask] == tgt_target[mask]).sum().item()
        total_tokens += mask.sum().item()

    if scheduler is not None:
        scheduler.step()

    avg_loss = total_loss / max(1, len(dataloader))
    accuracy = total_correct / max(1, total_tokens)

    return {"loss": avg_loss, "accuracy": accuracy}


# ──────────────────────────────────────────────────────────────
# Checkpointing
# ──────────────────────────────────────────────────────────────

def save_checkpoint(
    model: DeobfuscatorModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    is_best: bool = False,
):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }

    # Periodic checkpoint
    path = checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt"
    torch.save(state, path)

    # Best model
    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(state, best_path)

    return path


def load_checkpoint(
    model: DeobfuscatorModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: Path,
    device: torch.device,
) -> int:
    """Load a checkpoint and return the epoch number."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Train the Neural Code De-obfuscator",
    )
    parser.add_argument(
        "--data-dir", type=str, default="benchmarks/data/clean",
        help="Directory of clean .py files for training",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max-seq-len", type=int, default=1024, help="Max sequence length")
    parser.add_argument(
        "--checkpoint-dir", type=str, default="checkpoints",
        help="Directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Seed everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Dataset
    dataset = DeobfuscationDataset(
        data_dir=args.data_dir,
        max_seq_len=args.max_seq_len,
    )
    print(f"Loaded {len(dataset.sources)} clean files → {len(dataset)} training pairs")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Avoid multiprocessing issues with AST parsing
    )

    # Model
    feature_extractor = GraphFeatureExtractor(num_eigenvalues=32)
    model = DeobfuscatorModel(
        vocab_size=256,
        dim=256,
        num_heads=8,
        ffn_dim=1024,
        num_encoder_layers=6,
        num_decoder_layers=6,
        graph_feature_dim=feature_extractor.feature_dim,
        max_seq_len=args.max_seq_len,
    ).to(device)

    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # Resume from checkpoint if requested
    start_epoch = 0
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            start_epoch = load_checkpoint(model, optimizer, resume_path, device)
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"Warning: checkpoint {args.resume} not found, starting fresh")

    # Training
    checkpoint_dir = Path(args.checkpoint_dir)
    best_loss = float("inf")

    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        metrics = train(model, dataloader, optimizer, scheduler, device, epoch)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.3f} | "
            f"LR: {lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save checkpoint
        is_best = metrics["loss"] < best_loss
        if is_best:
            best_loss = metrics["loss"]

        # Save every 5 epochs + best model
        if (epoch + 1) % 5 == 0 or is_best or epoch == args.epochs - 1:
            path = save_checkpoint(
                model, optimizer, epoch + 1, metrics["loss"],
                checkpoint_dir, is_best=is_best,
            )
            tag = " ★ best" if is_best else ""
            print(f"  → Saved checkpoint: {path}{tag}")

    print("-" * 60)
    print(f"Training complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")


if __name__ == "__main__":
    main()

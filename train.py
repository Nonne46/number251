import argparse
import json
import os
import random

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Dataset

from models import SS13MapDiffusionLightning


class SS13MapDataset(Dataset):
    def __init__(self, hf_dataset):
        self.dataset = hf_dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Convert to numpy arrays
        tensor_data = np.array(item["tensor_data"], dtype=np.int64)
        tensor_mask = np.array(item["tensor_mask"], dtype=bool)

        return {
            "tensor_data": tensor_data,
            "tensor_mask": tensor_mask,
            "map_name": item["map_name"],
            "chunk_id": item["chunk_id"],
        }


def load_tokenizer_config(dataset_path):
    """Load tokenizer config from dataset directory"""
    config_path = os.path.join(dataset_path, "tokenizer_config.json")

    if os.path.exists(config_path):
        print(f"Loading tokenizer config from dataset: {config_path}")
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        print(f"Warning: No tokenizer config found in dataset directory")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train SS13 Map Diffusion Model")

    # Dataset arguments
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./ss13_map_dataset",
        help="Path to preprocessed dataset directory",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="tiles.json",
        help="Path to tokenizer JSON file (fallback if not in dataset)",
    )

    # Training arguments
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument(
        "--val_split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers"
    )

    # Model arguments
    parser.add_argument(
        "--base_channels", type=int, default=64, help="Base channels for U-Net"
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="Number of diffusion timesteps"
    )

    # Experiment arguments
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="ss13_diffusion",
        help="Name for this experiment",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument("--seed", type=int, default=31337, help="Random seed")

    # Parse arguments
    args = parser.parse_args()

    # Set random seeds
    pl.seed_everything(args.seed)

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    dataset = load_from_disk(args.dataset_path)

    # Try to load tokenizer config from dataset first
    dataset_tokenizer_config = load_tokenizer_config(args.dataset_path)

    if dataset_tokenizer_config:
        vocab_size = dataset_tokenizer_config["vocab_size"]
        max_layers = dataset_tokenizer_config["max_layers"]
        target_size = dataset_tokenizer_config.get("target_size", [16, 16])
    else:
        # Fallback to loading from tokenizer file
        print(f"Loading tokenizer from {args.tokenizer_path}...")
        with open(args.tokenizer_path, "r") as f:
            tokenizer_config = json.load(f)
        vocab_size = len(tokenizer_config["token_to_id"])
        max_layers = 16  # Default
        target_size = [16, 16]  # Default

    print(f"Vocabulary size: {vocab_size}")
    print(f"Max layers: {max_layers}")
    print(f"Target size: {target_size[0]}x{target_size[1]}")

    # Split dataset
    dataset_size = len(dataset)
    val_size = int(dataset_size * args.val_split)
    train_size = dataset_size - val_size

    # Shuffle and split
    dataset = dataset.shuffle(seed=args.seed)
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, dataset_size))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create PyTorch datasets
    train_pytorch_dataset = SS13MapDataset(train_dataset)
    val_pytorch_dataset = SS13MapDataset(val_dataset)

    # Create dataloaders
    train_loader = DataLoader(
        train_pytorch_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_pytorch_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # Initialize or load model
    if args.checkpoint:
        print(f"Loading model from checkpoint: {args.checkpoint}")
        model = SS13MapDiffusionLightning.load_from_checkpoint(
            args.checkpoint,
            vocab_size=vocab_size,
            layers=max_layers,
            base_channels=args.base_channels,
            timesteps=args.timesteps,
            learning_rate=args.learning_rate,
        )
    else:
        print("Initializing new model...")
        model = SS13MapDiffusionLightning(
            vocab_size=vocab_size,
            layers=max_layers,
            base_channels=args.base_channels,
            timesteps=args.timesteps,
            learning_rate=args.learning_rate,
        )

    # Create experiment directory
    experiment_dir = f"experiments/{args.experiment_name}"
    os.makedirs(experiment_dir, exist_ok=True)

    # Save training config
    training_config = {
        "dataset_path": args.dataset_path,
        "vocab_size": vocab_size,
        "max_layers": max_layers,
        "target_size": target_size,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_epochs": args.max_epochs,
        "timesteps": args.timesteps,
        "base_channels": args.base_channels,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "seed": args.seed,
    }

    with open(os.path.join(experiment_dir, "training_config.json"), "w") as f:
        json.dump(training_config, f, indent=2)

    # Logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.experiment_name,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, "checkpoints"),
        filename="epoch={epoch:02d}-val_loss={val_loss:.3f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
        every_n_epochs=1,
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss",
    #     patience=80,
    #     mode="min",
    #     verbose=False,
    # )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.5,
        logger=logger,
        enable_progress_bar=True,
        enable_checkpointing=True,
    )

    # Train
    print(f"\nStarting training for experiment: {args.experiment_name}")
    print(f"Logs will be saved to: logs/{args.experiment_name}")
    print(f"Checkpoints will be saved to: {experiment_dir}/checkpoints")

    if args.checkpoint:
        # Resume training from checkpoint
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.checkpoint)
    else:
        # Start fresh training
        trainer.fit(model, train_loader, val_loader)

    print(f"\nTraining complete! Results saved to {experiment_dir}")


if __name__ == "__main__":
    main()

import json
import random

import numpy as np
import pytorch_lightning as pl
import torch
from datasets import load_from_disk
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from models import SS13MapDiffusionLightning


class SS13MapDataset(Dataset):
    def __init__(self, hf_dataset, augment=True):
        self.dataset = hf_dataset
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Convert to numpy arrays
        tensor_data = np.array(item["tensor_data"], dtype=np.int64)
        tensor_mask = np.array(item["tensor_mask"], dtype=bool)

        # Apply augmentations
        # if self.augment and random.random() < 0.5:
        #     # Random rotation (0, 90, 180, 270 degrees)
        #     k = random.randint(0, 3)
        #     if k > 0:
        #         tensor_data = np.rot90(tensor_data, k, axes=(1, 2))
        #         tensor_mask = np.rot90(tensor_mask, k, axes=(1, 2))
        #
        #     # Random flips
        #     if random.random() < 0.5:
        #         tensor_data = np.flip(tensor_data, axis=1).copy()  # Vertical flip
        #         tensor_mask = np.flip(tensor_mask, axis=1).copy()
        #
        #     if random.random() < 0.5:
        #         tensor_data = np.flip(tensor_data, axis=2).copy()  # Horizontal flip
        #         tensor_mask = np.flip(tensor_mask, axis=2).copy()

        return {
            "tensor_data": tensor_data,
            "tensor_mask": tensor_mask,
            "map_name": item["map_name"],
            "chunk_id": item["chunk_id"],
        }


def main():
    # Configuration
    batch_size = 16  # Reduced for 16 layers
    num_epochs = 100
    learning_rate = 1e-4
    num_workers = 4
    val_split = 0.1

    # Load tokenizer config to get vocab size
    with open("tiles.json", "r") as f:
        tokenizer_config = json.load(f)
    vocab_size = len(tokenizer_config["token_to_id"])

    print(f"Vocabulary size: {vocab_size}")
    print(f"Max layers: 16")
    print(f"Target size: 16x16")

    # Load dataset
    print("Loading dataset...")
    dataset = load_from_disk("./dataset/")

    # Split dataset
    dataset_size = len(dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size

    # Shuffle and split
    dataset = dataset.shuffle(seed=31337)
    train_dataset = dataset.select(range(train_size))
    val_dataset = dataset.select(range(train_size, dataset_size))

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Create PyTorch datasets
    train_pytorch_dataset = SS13MapDataset(train_dataset, augment=True)
    val_pytorch_dataset = SS13MapDataset(val_dataset, augment=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_pytorch_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_pytorch_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Initialize model
    model = SS13MapDiffusionLightning(
        vocab_size=vocab_size,
        layers=16,
        base_channels=64,
        timesteps=1000,
        learning_rate=learning_rate,
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="ss13-diffusion-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        val_check_interval=0.5,  # Check validation 4 times per epoch
    )

    # Train
    print("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Test generation after training
    print("\nGenerating sample maps...")
    model.eval()
    with torch.no_grad():
        # Generate 4 16x16 maps
        generated = model.sample(
            shape=(4, 16, 16, 16), device=model.device  # (batch, layers, height, width)
        )

        print(f"Generated shape: {generated.shape}")
        print(f"Unique tokens in layer 0: {torch.unique(generated[:, 0])}")


if __name__ == "__main__":
    main()

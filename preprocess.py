import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
from datasets import Dataset
from tqdm import tqdm

from tokenizer import MapTensor, SS13MapTokenizer


@dataclass
class ChunkConfig:
    """Configuration for map chunking"""

    target_size: Tuple[int, int] = (16, 16)
    stride: Tuple[int, int] = (8, 8)  # Step size for sliding window
    min_content_ratio: float = 0.05  # Minimum non-empty content to keep chunk
    max_dominance_ratio: float = 0.8  # Max ratio for any single token
    min_variety: int = 3  # Minimum unique meaningful tokens
    augment: bool = True
    augment_variations: int = 4  # How many augmented versions per chunk


class SS13MapPreprocessor:
    def __init__(
        self,
        tokenizer_path: str,
        max_layers: int = 16,
        chunk_config: Optional[ChunkConfig] = None,
    ):
        """
        Initialize the preprocessor

        Args:
            tokenizer_path: Path to the tiles.json file
            max_layers: Maximum number of layers to use
            chunk_config: Configuration for chunking large maps
        """
        self.tokenizer = SS13MapTokenizer(tokenizer_path, max_layers=max_layers)
        self.chunk_config = chunk_config or ChunkConfig()

    def load_map_files(self, map_dir: str, pattern: str = "*.dmm") -> List[Path]:
        """Load all map files from directory"""
        map_path = Path(map_dir)
        if not map_path.exists():
            raise FileNotFoundError(f"Map directory not found: {map_dir}")

        map_files = list(map_path.glob(pattern))
        print(f"Found {len(map_files)} map files in {map_dir}")
        return map_files

    def calculate_chunks(
        self, map_shape: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Calculate chunk positions for a map using sliding window

        Returns:
            List of (start_y, end_y, start_x, end_x) tuples
        """
        height, width = map_shape
        target_h, target_w = self.chunk_config.target_size
        stride_h, stride_w = self.chunk_config.stride

        # If map is smaller than target, return single chunk (will be padded)
        if height <= target_h and width <= target_w:
            return [(0, height, 0, width)]

        chunks = []

        # Generate chunk positions with sliding window
        for y in range(0, height - target_h + 1, stride_h):
            for x in range(0, width - target_w + 1, stride_w):
                chunks.append((y, y + target_h, x, x + target_w))

        # Add edge chunks if there's remaining space
        # Right edge
        if width % stride_w != 0 and width > target_w:
            for y in range(0, height - target_h + 1, stride_h):
                chunks.append((y, y + target_h, width - target_w, width))

        # Bottom edge
        if height % stride_h != 0 and height > target_h:
            for x in range(0, width - target_w + 1, stride_w):
                chunks.append((height - target_h, height, x, x + target_w))

        # Bottom-right corner
        if height > target_h and width > target_w:
            chunks.append((height - target_h, height, width - target_w, width))

        # Remove duplicates
        chunks = list(set(chunks))

        return chunks

    def augment_chunk(
        self, data: np.ndarray, mask: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate augmented versions of a chunk

        Yields:
            Augmented (data, mask) tuples
        """
        # Original
        yield data, mask

        if not self.chunk_config.augment:
            return

        # Rotations (90, 180, 270 degrees)
        for k in [1, 2, 3]:
            rotated_data = np.rot90(data, k, axes=(1, 2))
            rotated_mask = np.rot90(mask, k, axes=(1, 2))
            yield rotated_data, rotated_mask

        # Flips (only if we want more variations)
        if self.chunk_config.augment_variations > 4:
            # Horizontal flip
            h_flip_data = np.flip(data, axis=2)
            h_flip_mask = np.flip(mask, axis=2)
            yield h_flip_data.copy(), h_flip_mask.copy()

            # Vertical flip
            v_flip_data = np.flip(data, axis=1)
            v_flip_mask = np.flip(mask, axis=1)
            yield v_flip_data.copy(), v_flip_mask.copy()

            # Both flips (equivalent to 180 rotation but different ordering)
            hv_flip_data = np.flip(np.flip(data, axis=1), axis=2)
            hv_flip_mask = np.flip(np.flip(mask, axis=1), axis=2)
            yield hv_flip_data.copy(), hv_flip_mask.copy()

    def extract_chunk(
        self, map_tensor, chunk_bounds: Tuple[int, int, int, int]
    ) -> Optional[MapTensor]:
        """
        Extract a chunk from the map tensor

        Args:
            map_tensor: Tokenized map tensor
            chunk_bounds: (start_y, end_y, start_x, end_x)

        Returns:
            Chunked and possibly padded tensor, or None if chunk has insufficient content
        """
        start_y, end_y, start_x, end_x = chunk_bounds

        # Extract chunk data
        chunk_data = map_tensor.data[:, start_y:end_y, start_x:end_x]
        chunk_mask = map_tensor.mask[:, start_y:end_y, start_x:end_x]

        # Check variety in layer 0 (turfs) - this is the most important layer
        turf_layer = chunk_data[0]
        turf_mask = chunk_mask[0]

        if np.sum(turf_mask) == 0:
            return None

        # Count occurrences of each token in the turf layer
        unique_tokens, counts = np.unique(turf_layer[turf_mask], return_counts=True)
        total_positions = np.sum(turf_mask)

        # Find the most common token and its ratio
        max_count = np.max(counts)
        dominance_ratio = max_count / total_positions

        # Reject if one token dominates too much
        if dominance_ratio > self.chunk_config.max_dominance_ratio:
            return None

        # Also check if there's enough variety across all layers
        all_tokens = []
        for layer in range(len(chunk_data)):
            layer_tokens = chunk_data[layer][chunk_mask[layer]]
            # Exclude empty and pad tokens from variety count
            meaningful_tokens = layer_tokens[
                (layer_tokens != self.tokenizer.EMPTY_ID)
                & (layer_tokens != self.tokenizer.PAD_ID)
            ]
            all_tokens.extend(meaningful_tokens)

        # Need at least some minimum variety
        unique_meaningful = len(np.unique(all_tokens))
        if unique_meaningful < self.chunk_config.min_variety:
            return None

        # Create new tensor with chunk data
        chunk_shape = (end_y - start_y, end_x - start_x)

        # If chunk is smaller than target size, pad it
        if chunk_shape != self.chunk_config.target_size:
            # Create padded arrays
            target_h, target_w = self.chunk_config.target_size
            padded_data = np.full(
                (self.tokenizer.max_layers, target_h, target_w),
                self.tokenizer.PAD_ID,
                dtype=chunk_data.dtype,
            )
            padded_mask = np.zeros(
                (self.tokenizer.max_layers, target_h, target_w), dtype=bool
            )

            # Copy chunk data to padded arrays
            h, w = chunk_shape
            padded_data[:, :h, :w] = chunk_data
            padded_mask[:, :h, :w] = chunk_mask

            return MapTensor(
                data=padded_data,
                mask=padded_mask,
                original_shape=chunk_shape,
                padding=(0, target_h - h, 0, target_w - w),
            )
        else:
            # No padding needed
            return MapTensor(
                data=chunk_data.copy(),
                mask=chunk_mask.copy(),
                original_shape=chunk_shape,
                padding=(0, 0, 0, 0),
            )

    def process_single_map(self, map_path: Path) -> Iterator[Dict]:
        """
        Process a single map file and yield training examples

        Args:
            map_path: Path to the .dmm file

        Yields:
            Dictionary containing processed map data
        """
        try:
            # Read map file
            with open(map_path, "r", encoding="utf-8") as f:
                map_content = f.read()

            # Tokenize the map
            map_tensor = self.tokenizer.tokenize_map(map_content)

            # Calculate chunks
            chunks = self.calculate_chunks(map_tensor.original_shape)

            # Process each chunk
            chunk_id = 0
            valid_chunks = 0

            for chunk_bounds in chunks:
                chunk_tensor = self.extract_chunk(map_tensor, chunk_bounds)

                if chunk_tensor is None:
                    continue

                valid_chunks += 1

                # Generate augmented versions
                for aug_idx, (aug_data, aug_mask) in enumerate(
                    self.augment_chunk(chunk_tensor.data, chunk_tensor.mask)
                ):
                    # Stop if we've generated enough variations
                    if aug_idx >= self.chunk_config.augment_variations:
                        break

                    # Create training example
                    example = {
                        "map_name": map_path.stem,
                        "chunk_id": chunk_id,
                        "augmentation_id": aug_idx,
                        "chunk_bounds": chunk_bounds,
                        "original_shape": list(map_tensor.original_shape),
                        "tensor_data": aug_data.tolist(),
                        "tensor_mask": aug_mask.tolist(),
                        "padding": chunk_tensor.padding,
                    }

                    yield example
                    chunk_id += 1

            # Log if we filtered out many chunks
            if valid_chunks < len(chunks) * 0.1:  # Less than 10% valid
                print(
                    f"  Warning: {map_path.name} had only {valid_chunks}/{len(chunks)} valid chunks"
                )

        except Exception as e:
            print(f"Error processing {map_path}: {e}")
            import traceback

            traceback.print_exc()

    def process_maps(self, map_dir: str, output_dir: str = "./ss13_map_dataset"):
        """
        Process all maps and create HuggingFace dataset

        Args:
            map_dir: Directory containing .dmm files
            output_dir: Output directory for processed dataset
        """
        # Load map files
        map_files = self.load_map_files(map_dir)

        if not map_files:
            print("No map files found!")
            return None

        # Process all maps
        all_examples = []

        print("Processing maps...")
        print(
            f"Chunk config: size={self.chunk_config.target_size}, stride={self.chunk_config.stride}"
        )
        print(
            f"Filtering: max_dominance={self.chunk_config.max_dominance_ratio}, min_variety={self.chunk_config.min_variety}"
        )

        for map_path in tqdm(map_files, desc="Processing maps"):
            map_examples = list(self.process_single_map(map_path))
            all_examples.extend(map_examples)

            # Print progress info
            if len(map_examples) > 0:
                print(f"  {map_path.name}: {len(map_examples)} examples")

        print(f"\nGenerated {len(all_examples)} total training examples")

        if not all_examples:
            print("No valid examples generated!")
            return None

        # Create HuggingFace dataset
        dataset = Dataset.from_list(all_examples)

        # Save dataset
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)

        # Save tokenizer config
        tokenizer_config = {
            "vocab_size": self.tokenizer.get_vocab_size(),
            "max_layers": self.tokenizer.max_layers,
            "target_size": list(self.chunk_config.target_size),
            "stride": list(self.chunk_config.stride),
            "max_dominance_ratio": self.chunk_config.max_dominance_ratio,
            "min_variety": self.chunk_config.min_variety,
            "augmentations_per_chunk": self.chunk_config.augment_variations,
            "reserved_tokens": {
                "EMPTY_ID": self.tokenizer.EMPTY_ID,
                "UNK_ID": self.tokenizer.UNK_ID,
                "PAD_ID": self.tokenizer.PAD_ID,
                "MASK_ID": self.tokenizer.MASK_ID,
            },
        }

        with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        print(f"\nDataset saved to {output_dir}")
        print(f"Total examples: {len(dataset)}")

        # Print some statistics
        print("\nDataset statistics:")
        print(f"  Unique maps: {len(set(ex['map_name'] for ex in all_examples))}")
        print(f"  Average examples per map: {len(all_examples) / len(map_files):.1f}")

        return dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess SS13 maps for training")
    parser.add_argument(
        "--map_dir", required=True, help="Directory containing .dmm files"
    )
    parser.add_argument("--tokenizer_path", required=True, help="Path to tiles.json")
    parser.add_argument(
        "--output_dir", default="./ss13_map_dataset", help="Output directory"
    )
    parser.add_argument("--max_layers", type=int, default=16, help="Maximum layers")
    parser.add_argument(
        "--target_size", nargs=2, type=int, default=[16, 16], help="Target chunk size"
    )
    parser.add_argument(
        "--stride", nargs=2, type=int, default=[8, 8], help="Stride for sliding window"
    )
    parser.add_argument(
        "--min_content",
        type=float,
        default=0.05,
        help="Minimum content ratio for chunks",
    )
    parser.add_argument(
        "--max_dominance",
        type=float,
        default=0.8,
        help="Maximum dominance ratio for single token",
    )
    parser.add_argument(
        "--min_variety", type=int, default=3, help="Minimum variety of tokens"
    )
    parser.add_argument(
        "--no_augment", action="store_true", help="Disable augmentation"
    )
    parser.add_argument(
        "--augment_variations",
        type=int,
        default=4,
        help="Number of augmented versions per chunk",
    )

    args = parser.parse_args()

    # Create chunk config
    chunk_config = ChunkConfig(
        target_size=tuple(args.target_size),
        stride=tuple(args.stride),
        min_content_ratio=args.min_content,
        max_dominance_ratio=args.max_dominance,
        min_variety=args.min_variety,
        augment=not args.no_augment,
        augment_variations=args.augment_variations,
    )

    # Create preprocessor
    preprocessor = SS13MapPreprocessor(
        tokenizer_path=args.tokenizer_path,
        max_layers=args.max_layers,
        chunk_config=chunk_config,
    )

    # Process maps
    dataset = preprocessor.process_maps(args.map_dir, args.output_dir)

    if dataset:
        print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()

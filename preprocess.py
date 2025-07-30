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
    max_dominance_ratio: float = 0.8  # Max ratio for any single token
    min_variety: int = 3  # Minimum unique meaningful tokens
    augment: bool = False  # Simple augmentation: 3 rotations + 2 flips


class SS13MapPreprocessor:
    def __init__(
        self,
        tokenizer_path: str,
        max_layers: int = 16,
        chunk_config: Optional[ChunkConfig] = None,
    ):
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
        Calculate chunk positions using sliding window with half-target stride
        Returns: List of (start_y, end_y, start_x, end_x) tuples
        """
        height, width = map_shape
        target_h, target_w = self.chunk_config.target_size

        # Stride is half of target size for 50% overlap
        stride_h = target_h // 2
        stride_w = target_w // 2

        # If map is smaller than target, return single chunk (will be padded)
        if height <= target_h and width <= target_w:
            return [(0, height, 0, width)]

        chunks = []

        # Generate sliding window positions
        y = 0
        while y + target_h <= height:
            x = 0
            while x + target_w <= width:
                chunks.append((y, y + target_h, x, x + target_w))
                x += stride_w
            y += stride_h

        # Add edge chunks to ensure full coverage
        # Right edge
        if width > target_w:
            y = 0
            while y + target_h <= height:
                chunks.append((y, y + target_h, width - target_w, width))
                y += stride_h

        # Bottom edge
        if height > target_h:
            x = 0
            while x + target_w <= width:
                chunks.append((height - target_h, height, x, x + target_w))
                x += stride_w

        # Bottom-right corner
        if height > target_h and width > target_w:
            chunks.append((height - target_h, height, width - target_w, width))

        # Remove duplicates
        return list(set(chunks))

    def is_chunk_meaningful(
        self, chunk_data: np.ndarray, chunk_mask: np.ndarray, is_small_map: bool = False
    ) -> bool:
        """
        Check if chunk has meaningful content and isn't dominated by single token

        Args:
            chunk_data: Chunk tensor data
            chunk_mask: Chunk mask
            is_small_map: True if original map is smaller than target size
        """
        # For small maps, we're more lenient about empty space dominance
        max_dominance = 0.9 if is_small_map else self.chunk_config.max_dominance_ratio

        # Check variety in layer 0 (turfs) - most important layer
        turf_layer = chunk_data[0]
        turf_mask = chunk_mask[0]

        if np.sum(turf_mask) == 0:
            return False

        # Count token occurrences in turf layer
        unique_tokens, counts = np.unique(turf_layer[turf_mask], return_counts=True)
        total_positions = np.sum(turf_mask)

        # Check dominance ratio
        max_count = np.max(counts)
        dominance_ratio = max_count / total_positions

        if dominance_ratio > max_dominance:
            return False

        # Check variety across all layers
        all_meaningful_tokens = []
        for layer in range(len(chunk_data)):
            layer_tokens = chunk_data[layer][chunk_mask[layer]]
            # Exclude empty and pad tokens
            meaningful_tokens = layer_tokens[
                (layer_tokens != self.tokenizer.EMPTY_ID)
                & (layer_tokens != self.tokenizer.PAD_ID)
            ]
            all_meaningful_tokens.extend(meaningful_tokens)

        unique_meaningful = len(np.unique(all_meaningful_tokens))
        return unique_meaningful >= self.chunk_config.min_variety

    def extract_chunk(
        self, map_tensor: MapTensor, chunk_bounds: Tuple[int, int, int, int]
    ) -> Optional[MapTensor]:
        """Extract and validate a chunk from the map tensor"""
        start_y, end_y, start_x, end_x = chunk_bounds

        # Extract chunk
        chunk_data = map_tensor.data[:, start_y:end_y, start_x:end_x]
        chunk_mask = map_tensor.mask[:, start_y:end_y, start_x:end_x]

        # Check if original map is small
        is_small_map = (
            map_tensor.original_shape[0] <= self.chunk_config.target_size[0]
            and map_tensor.original_shape[1] <= self.chunk_config.target_size[1]
        )

        # Validate chunk meaningfulness
        if not self.is_chunk_meaningful(chunk_data, chunk_mask, is_small_map):
            return None

        # Pad if necessary
        chunk_shape = (end_y - start_y, end_x - start_x)
        target_h, target_w = self.chunk_config.target_size

        if chunk_shape != self.chunk_config.target_size:
            # Create padded arrays
            padded_data = np.full(
                (self.tokenizer.max_layers, target_h, target_w),
                self.tokenizer.PAD_ID,
                dtype=chunk_data.dtype,
            )
            padded_mask = np.zeros(
                (self.tokenizer.max_layers, target_h, target_w), dtype=bool
            )

            # Copy chunk data
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
            return MapTensor(
                data=chunk_data.copy(),
                mask=chunk_mask.copy(),
                original_shape=chunk_shape,
                padding=(0, 0, 0, 0),
            )

    def augment_chunk(
        self, data: np.ndarray, mask: np.ndarray
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate augmented versions of a chunk"""
        # Original
        yield data, mask

        if not self.chunk_config.augment:
            return

        # 3 Rotations (90, 180, 270 degrees)
        for k in [1, 2, 3]:
            rotated_data = np.rot90(data, k, axes=(1, 2))
            rotated_mask = np.rot90(mask, k, axes=(1, 2))
            yield rotated_data, rotated_mask

        # Vertical flip
        v_flip_data = np.flip(data, axis=1)
        v_flip_mask = np.flip(mask, axis=1)
        yield v_flip_data.copy(), v_flip_mask.copy()

        # Horizontal flip
        h_flip_data = np.flip(data, axis=2)
        h_flip_mask = np.flip(mask, axis=2)
        yield h_flip_data.copy(), h_flip_mask.copy()

    def process_single_map(self, map_path: Path) -> Iterator[Dict]:
        """Process a single map file and yield training examples"""
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                map_content = f.read()

            # Tokenize map
            map_tensor = self.tokenizer.tokenize_map(map_content)

            # Calculate chunks
            chunks = self.calculate_chunks(map_tensor.original_shape)

            chunk_id = 0
            valid_chunks = 0

            for chunk_bounds in chunks:
                chunk_tensor = self.extract_chunk(map_tensor, chunk_bounds)

                if chunk_tensor is None:
                    continue

                valid_chunks += 1

                # Generate augmented versions (up to 6 total: original + 3 rotations + 2 flips)
                for aug_idx, (aug_data, aug_mask) in enumerate(
                    self.augment_chunk(chunk_tensor.data, chunk_tensor.mask)
                ):

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

            if valid_chunks < len(chunks) * 0.1:  # Less than 10% valid
                print(
                    f"  Warning: {map_path.name} had only {valid_chunks}/{len(chunks)} valid chunks"
                )

        except Exception as e:
            print(f"Error processing {map_path}: {e}")

    def process_maps(self, map_dir: str, output_dir: str = "./ss13_map_dataset"):
        """Process all maps and create HuggingFace dataset"""
        map_files = self.load_map_files(map_dir)

        if not map_files:
            print("No map files found!")
            return None

        all_examples = []
        target_h, target_w = self.chunk_config.target_size

        print(
            f"Processing maps with {target_h}x{target_w} chunks, {target_h//2}x{target_w//2} stride"
        )
        print(
            f"Filtering: max_dominance={self.chunk_config.max_dominance_ratio}, min_variety={self.chunk_config.min_variety}"
        )

        for map_path in tqdm(map_files, desc="Processing maps"):
            map_examples = list(self.process_single_map(map_path))
            all_examples.extend(map_examples)

            if len(map_examples) > 0:
                print(f"  {map_path.name}: {len(map_examples)} examples")

        print(f"\nGenerated {len(all_examples)} total training examples")

        if not all_examples:
            print("No valid examples generated!")
            return None

        # Create and save dataset
        dataset = Dataset.from_list(all_examples)
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)

        # Save tokenizer config
        augment_count = (
            6 if self.chunk_config.augment else 1
        )  # 1 original + 3 rotations + 2 flips
        tokenizer_config = {
            "vocab_size": self.tokenizer.get_vocab_size(),
            "max_layers": self.tokenizer.max_layers,
            "target_size": list(self.chunk_config.target_size),
            "stride": [s // 2 for s in self.chunk_config.target_size],  # Half of target
            "max_dominance_ratio": self.chunk_config.max_dominance_ratio,
            "min_variety": self.chunk_config.min_variety,
            "augment": self.chunk_config.augment,
            "augmentations_per_chunk": augment_count,
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
        print(f"Unique maps: {len(set(ex['map_name'] for ex in all_examples))}")
        print(f"Average examples per map: {len(all_examples) / len(map_files):.1f}")

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
        "--augment",
        action="store_true",
        help="Enable augmentation (3 rotations + 2 flips)",
    )
    parser.add_argument(
        "--max_dominance", type=float, default=0.8, help="Maximum dominance ratio"
    )
    parser.add_argument(
        "--min_variety", type=int, default=3, help="Minimum variety of tokens"
    )

    args = parser.parse_args()

    chunk_config = ChunkConfig(
        target_size=tuple(args.target_size),
        max_dominance_ratio=args.max_dominance,
        min_variety=args.min_variety,
        augment=args.augment,
    )

    preprocessor = SS13MapPreprocessor(
        tokenizer_path=args.tokenizer_path,
        max_layers=args.max_layers,
        chunk_config=chunk_config,
    )

    dataset = preprocessor.process_maps(args.map_dir, args.output_dir)

    if dataset:
        print("\nPreprocessing complete!")


if __name__ == "__main__":
    main()

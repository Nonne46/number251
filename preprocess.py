import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

# HuggingFace datasets
from datasets import Dataset
from tqdm import tqdm

# Import your tokenizer (assuming it's in the same directory)
from tokenizer import SS13MapTokenizer


@dataclass
class ChunkConfig:
    """Configuration for map chunking"""

    target_size: Tuple[int, int] = (10, 10)
    overlap_percent: float = 0.2  # 20% overlap between chunks
    min_content_ratio: float = 0.1  # Minimum non-empty content to keep chunk


class SS13MapPreprocessor:
    def __init__(
        self,
        tokenizer_path: str,
        max_layers: int = 8,
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
        Calculate chunk positions for a map

        Returns:
            List of (start_y, end_y, start_x, end_x) tuples
        """
        height, width = map_shape
        target_h, target_w = self.chunk_config.target_size

        # If map is smaller than target, return single chunk (will be padded)
        if height <= target_h and width <= target_w:
            return [(0, height, 0, width)]

        chunks = []
        overlap_h = int(target_h * self.chunk_config.overlap_percent)
        overlap_w = int(target_w * self.chunk_config.overlap_percent)

        step_h = target_h - overlap_h
        step_w = target_w - overlap_w

        # Generate chunk positions
        for y in range(0, height, step_h):
            for x in range(0, width, step_w):
                end_y = min(y + target_h, height)
                end_x = min(x + target_w, width)

                # Adjust start position if we're at the edge
                start_y = max(0, end_y - target_h)
                start_x = max(0, end_x - target_w)

                chunks.append((start_y, end_y, start_x, end_x))

        return chunks

    def extract_chunk(
        self, map_tensor, chunk_bounds: Tuple[int, int, int, int]
    ) -> Optional[object]:
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

        # Check content ratio
        total_positions = np.sum(chunk_mask)
        non_empty_positions = np.sum(
            (chunk_data != self.tokenizer.EMPTY_ID)
            & (chunk_data != self.tokenizer.PAD_ID)
            & chunk_mask
        )

        if (
            total_positions == 0
            or (non_empty_positions / total_positions)
            < self.chunk_config.min_content_ratio
        ):
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

            # Create padded tensor object
            from tokenizer import MapTensor  # Assuming this exists in your tokenizer

            return MapTensor(
                data=padded_data,
                mask=padded_mask,
                original_shape=chunk_shape,
                padding=(0, target_h - h, 0, target_w - w),
            )
        else:
            # No padding needed
            from tokenizer import MapTensor

            return MapTensor(
                data=chunk_data,
                mask=chunk_mask,
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
            for i, chunk_bounds in enumerate(chunks):
                chunk_tensor = self.extract_chunk(map_tensor, chunk_bounds)

                if chunk_tensor is None:
                    continue

                # Create training example
                example = {
                    "map_name": map_path.stem,
                    "chunk_id": i,
                    "chunk_bounds": chunk_bounds,
                    "original_shape": map_tensor.original_shape,
                    "tensor_data": chunk_tensor.data.tolist(),  # Convert to list for JSON serialization
                    "tensor_mask": chunk_tensor.mask.tolist(),
                    "padding": chunk_tensor.padding,
                }

                yield example

        except Exception as e:
            print(f"Error processing {map_path}: {e}")

    def process_maps(self, map_dir: str, output_dir: str = "./processed_maps"):
        """
        Process all maps and create HuggingFace dataset

        Args:
            map_dir: Directory containing .dmm files
            output_dir: Output directory for processed dataset
        """
        # Load map files
        map_files = self.load_map_files(map_dir)

        # Process all maps
        all_examples = []

        print("Processing maps...")
        for map_path in tqdm(map_files, desc="Processing maps"):
            for example in self.process_single_map(map_path):
                all_examples.append(example)

        print(f"Generated {len(all_examples)} training examples")

        # Create HuggingFace dataset
        dataset = Dataset.from_list(all_examples)

        # Save dataset
        os.makedirs(output_dir, exist_ok=True)
        dataset.save_to_disk(output_dir)

        # Save tokenizer config
        tokenizer_config = {
            "vocab_size": self.tokenizer.get_vocab_size(),
            "max_layers": self.tokenizer.max_layers,
            "target_size": self.chunk_config.target_size,
            "reserved_tokens": {
                "EMPTY_ID": self.tokenizer.EMPTY_ID,
                "UNK_ID": self.tokenizer.UNK_ID,
                "PAD_ID": self.tokenizer.PAD_ID,
                "MASK_ID": self.tokenizer.MASK_ID,
            },
        }

        with open(os.path.join(output_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f, indent=2)

        print(f"Dataset saved to {output_dir}")
        print(f"Total examples: {len(dataset)}")

        return dataset


def main():
    parser = argparse.ArgumentParser(description="Preprocess SS13 maps for training")
    parser.add_argument(
        "--map_dir", required=True, help="Directory containing .dmm files"
    )
    parser.add_argument("--tokenizer_path", required=True, help="Path to tiles.json")
    parser.add_argument(
        "--output_dir", default="./processed_maps", help="Output directory"
    )
    parser.add_argument("--max_layers", type=int, default=8, help="Maximum layers")
    parser.add_argument(
        "--target_size", nargs=2, type=int, default=[10, 10], help="Target chunk size"
    )
    parser.add_argument(
        "--overlap", type=float, default=0.2, help="Overlap percentage for chunking"
    )
    parser.add_argument(
        "--min_content",
        type=float,
        default=0.1,
        help="Minimum content ratio for chunks",
    )

    args = parser.parse_args()

    # Create chunk config
    chunk_config = ChunkConfig(
        target_size=tuple(args.target_size),
        overlap_percent=args.overlap,
        min_content_ratio=args.min_content,
    )

    # Create preprocessor
    preprocessor = SS13MapPreprocessor(
        tokenizer_path=args.tokenizer_path,
        max_layers=args.max_layers,
        chunk_config=chunk_config,
    )

    # Process maps
    dataset = preprocessor.process_maps(args.map_dir, args.output_dir)

    print("Preprocessing complete!")


if __name__ == "__main__":
    main()

import argparse
import json
import string
from pathlib import Path

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm

from tokenizer import SS13MapTokenizer


def tensor_to_dmm(tensor_map, tensor_mask, tokenizer, z_level=1):
    """Convert tensor map back to DMM format"""
    layers, height, width = tensor_map.shape

    # Dictionary to store unique tile combinations
    tile_combinations = {}
    combination_to_letter = {}
    letter_index = 0
    available_letters = list(string.ascii_letters)  # a-z, A-Z

    # Grid to store letter codes
    grid = [[None for _ in range(width)] for _ in range(height)]

    # Process each position
    for y in range(height):
        for x in range(width):
            # Skip if this position is padding (mask is False)
            if not tensor_mask[0, y, x]:
                continue

            # Get all objects at this position
            objects_at_pos = []

            # First, get the turf (should be in layer 0)
            turf_token_id = int(tensor_map[0, y, x])

            if turf_token_id != tokenizer.EMPTY_ID:
                turf_token = tokenizer.id_to_token.get(
                    turf_token_id, f"<UNK:{turf_token_id}>"
                )
                # Only add if it's a valid turf
                if turf_token.startswith("/turf/"):
                    objects_at_pos.append(turf_token)

            # Then add other objects from higher layers
            for layer in range(1, layers):
                token_id = int(tensor_map[layer, y, x])

                # Skip empty tokens
                if token_id == tokenizer.EMPTY_ID:
                    continue

                # Get token string
                token = tokenizer.id_to_token.get(token_id, f"<UNK:{token_id}>")

                # Skip special tokens and areas
                if token in ["<PAD>", "<MASK>", "<UNK>", "<EMPTY>"] or token.startswith(
                    "/area/"
                ):
                    continue

                objects_at_pos.append(token)

            # If no turf, use default plating
            if not any(obj.startswith("/turf/") for obj in objects_at_pos):
                objects_at_pos.insert(0, "/turf/open/floor/plating")

            # Add area to the turf line (always at the end)
            objects_at_pos.append("/area/template_noop")

            # Create tuple for unique combination
            combination = tuple(objects_at_pos)

            # Get or create letter for this combination
            if combination not in combination_to_letter:
                if letter_index < len(available_letters):
                    letter = available_letters[letter_index]
                else:
                    # If we run out of letters, use multi-character codes
                    letter = f"X{letter_index - len(available_letters)}"

                combination_to_letter[combination] = letter
                tile_combinations[letter] = combination
                letter_index += 1

            grid[y][x] = combination_to_letter[combination]

    # Build DMM string
    dmm_lines = []
    dmm_lines.append("//RECONSTRUCTED FROM DATASET - THIS IS A TEST FILE")

    # Write tile definitions - sorted for consistency
    for letter in sorted(tile_combinations.keys()):
        objects = tile_combinations[letter]
        dmm_lines.append(f'"{letter}" = (')

        for i, obj in enumerate(objects):
            if i == len(objects) - 1:
                dmm_lines.append(f"{obj})")  # Last item, close parenthesis
            else:
                dmm_lines.append(f"{obj},")  # Add comma for non-last items

    # Find actual bounds (non-padding area)
    min_x, max_x = width, -1
    min_y, max_y = height, -1

    for y in range(height):
        for x in range(width):
            if tensor_mask[0, y, x] and grid[y][x] is not None:
                min_x = min(min_x, x)
                max_x = max(max_x, x)
                min_y = min(min_y, y)
                max_y = max(max_y, y)

    # Write grid (only non-padding area)
    if min_x <= max_x and min_y <= max_y:
        dmm_lines.append(f'(1,1,{z_level}) = {{"')
        for y in range(min_y, max_y + 1):
            row = ""
            for x in range(min_x, max_x + 1):
                if grid[y][x] is not None:
                    row += grid[y][x]
                else:
                    row += " "  # Space for padding
            dmm_lines.append(row)
        dmm_lines.append('"}')

    return "\n".join(dmm_lines)


def visualize_chunk_info(item, tokenizer):
    """Print information about a chunk"""
    print(f"\n{'='*60}")
    print(f"Map: {item['map_name']}")
    print(f"Chunk ID: {item['chunk_id']}")
    print(f"Original shape: {item['original_shape']}")
    print(f"Chunk bounds: {item['chunk_bounds']}")
    print(f"Padding: {item['padding']}")

    tensor_data = np.array(item["tensor_data"])
    tensor_mask = np.array(item["tensor_mask"])

    print(f"Tensor shape: {tensor_data.shape}")

    # Count non-empty tokens per layer
    for layer in range(tensor_data.shape[0]):
        layer_data = tensor_data[layer]
        layer_mask = tensor_mask[layer]

        # Get unique tokens in valid area
        valid_tokens = layer_data[layer_mask]
        unique_tokens = np.unique(valid_tokens)

        non_empty = unique_tokens[unique_tokens != tokenizer.EMPTY_ID]
        if len(non_empty) > 0:
            print(f"\nLayer {layer}: {len(non_empty)} unique tokens")
            for token_id in non_empty[:5]:  # Show first 5
                token = tokenizer.id_to_token.get(int(token_id), f"<UNK:{token_id}>")
                print(f"  - {token}")
            if len(non_empty) > 5:
                print(f"  ... and {len(non_empty) - 5} more")


def main():
    parser = argparse.ArgumentParser(
        description="Test dataset by converting chunks back to DMM"
    )
    parser.add_argument(
        "--dataset", type=str, default="ss13_map_dataset", help="Dataset path"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="tiles.json", help="Tokenizer config path"
    )
    parser.add_argument(
        "--output-dir", type=str, default="test_chunks", help="Output directory"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10, help="Number of samples to convert"
    )
    parser.add_argument("--start-idx", type=int, default=0, help="Starting index")
    parser.add_argument(
        "--verbose", action="store_true", help="Print chunk information"
    )

    args = parser.parse_args()

    # Load tokenizer
    print(f"Loading tokenizer from {args.tokenizer}...")
    tokenizer = SS13MapTokenizer(args.tokenizer, max_layers=16)
    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")

    # Load dataset
    print(f"Loading dataset from {args.dataset}...")
    dataset = load_from_disk(args.dataset)
    print(f"Dataset size: {len(dataset)}")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    # Process samples
    end_idx = min(args.start_idx + args.num_samples, len(dataset))
    print(f"\nProcessing samples {args.start_idx} to {end_idx-1}...")

    for idx in tqdm(range(args.start_idx, end_idx), desc="Converting chunks"):
        item = dataset[idx]

        # Convert to numpy arrays
        tensor_data = np.array(item["tensor_data"], dtype=np.int64)
        tensor_mask = np.array(item["tensor_mask"], dtype=bool)

        # Print info if verbose
        if args.verbose:
            visualize_chunk_info(item, tokenizer)

        # Convert to DMM
        dmm_content = tensor_to_dmm(tensor_data, tensor_mask, tokenizer)

        # Save file
        filename = f"{item['map_name']}_chunk{item['chunk_id']:04d}.dmm"
        # Replace any path separators in map name
        filename = filename.replace("/", "_").replace("\\", "_")

        file_path = output_path / filename
        with open(file_path, "w") as f:
            f.write(dmm_content)

    print(
        f"\nConverted {end_idx - args.start_idx} chunks to DMM files in {output_path}"
    )

    # Show example of first converted file
    if end_idx > args.start_idx:
        first_file = list(output_path.glob("*.dmm"))[0]
        print(f"\nExample output ({first_file.name}):")
        print("-" * 60)
        with open(first_file, "r") as f:
            content = f.read()
            print(content[:1000] + "..." if len(content) > 1000 else content)


if __name__ == "__main__":
    main()

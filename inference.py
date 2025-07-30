import argparse
import json
import string
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models import SS13MapDiffusionLightning
from tokenizer import SS13MapTokenizer


class SS13MapGenerator:
    def __init__(self, checkpoint_path, tokenizer_path):
        """Initialize generator with checkpoint and tokenizer"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load tokenizer
        self.tokenizer = SS13MapTokenizer(tokenizer_path, max_layers=16)

        # Load model from checkpoint
        print(f"Loading model from {checkpoint_path}...")
        self.model = SS13MapDiffusionLightning.load_from_checkpoint(
            checkpoint_path, map_location=self.device
        )
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully on {self.device}")
        print(f"Vocabulary size: {self.model.vocab_size}")
        print(f"Layers: {self.model.hparams.layers}")

    @torch.no_grad()
    def generate_maps(
        self, num_maps=1, width=16, height=16, temperature=1.0, show_progress=True
    ):
        """Generate maps using the diffusion model"""
        print(f"\nGenerating {num_maps} maps of size {width}x{height}...")

        batch_size = min(num_maps, 4)  # Generate in batches of 4
        all_maps = []

        for batch_start in range(0, num_maps, batch_size):
            current_batch_size = min(batch_size, num_maps - batch_start)

            # Generate batch
            maps = self._sample_batch(
                current_batch_size,
                width,
                height,
                temperature,
                show_progress=show_progress,
            )
            all_maps.extend(maps)

        return all_maps

    def _sample_batch(
        self, batch_size, width, height, temperature=1.0, show_progress=True
    ):
        """Sample a batch of maps using DDPM"""
        layers = self.model.hparams.layers
        timesteps = self.model.timesteps

        # Start from pure noise
        x = torch.randint(
            0,
            self.model.vocab_size,
            (batch_size, layers, height, width),
            device=self.device,
        )

        # Create progress bar for denoising steps
        pbar = tqdm(
            reversed(range(timesteps)), desc="Denoising", disable=not show_progress
        )

        # Denoise step by step
        for t in pbar:
            t_batch = torch.full((batch_size,), t, device=self.device)

            # Predict denoised tokens
            logits = self.model(x, t_batch)  # (batch, layers, vocab, h, w)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=2)

            # Sample from distribution
            x_pred = torch.multinomial(
                probs.permute(0, 1, 3, 4, 2).reshape(-1, self.model.vocab_size),
                num_samples=1,
            ).reshape(batch_size, layers, height, width)

            # Mix with less noise for next step
            if t > 0:
                alpha_t = self.model.alphas_cumprod[t]
                alpha_prev = self.model.alphas_cumprod[t - 1]

                # Probability of replacing with predicted vs keeping noisy
                replace_prob = (alpha_t - alpha_prev) / alpha_t
                replace_mask = torch.rand_like(x, dtype=torch.float32) < replace_prob

                x = torch.where(replace_mask, x_pred, x)
            else:
                x = x_pred

            pbar.set_postfix({"t": t, "unique_tokens": len(torch.unique(x))})

        for i in range(batch_size):
            x[i] = self.postprocess_map(x[i])

        # Convert to numpy
        maps_np = x.cpu().numpy()
        return [maps_np[i] for i in range(batch_size)]

    def tensor_to_dmm(self, tensor_map, map_name="generated", z_level=1):
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
                # Get all objects at this position
                objects_at_pos = []

                # First, get the turf (should be in layer 0)
                turf_token_id = int(tensor_map[0, y, x])
                turf_token = None

                if turf_token_id != self.tokenizer.EMPTY_ID:
                    turf_token = self.tokenizer.id_to_token.get(
                        turf_token_id, f"<UNK:{turf_token_id}>"
                    )
                    # Only add if it's a valid turf
                    if turf_token.startswith("/turf/"):
                        objects_at_pos.append(turf_token)

                # Then add other objects from higher layers
                for layer in range(1, layers):
                    token_id = int(tensor_map[layer, y, x])

                    # Skip empty tokens
                    if token_id == self.tokenizer.EMPTY_ID:
                        continue

                    # Get token string
                    token = self.tokenizer.id_to_token.get(
                        token_id, f"<UNK:{token_id}>"
                    )

                    # Skip special tokens and areas (areas are part of turf definition)
                    if token in [
                        "<PAD>",
                        "<MASK>",
                        "<UNK>",
                        "<EMPTY>",
                    ] or token.startswith("/area/"):
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
        dmm_lines.append(
            "//MAP CONVERTED BY inference.py THIS HEADER COMMENT PREVENTS RECONVERSION, DO NOT REMOVE"
        )

        # Write tile definitions - sorted for consistency
        for letter in sorted(tile_combinations.keys()):
            objects = tile_combinations[letter]
            dmm_lines.append(f'"{letter}" = (')

            for i, obj in enumerate(objects):
                if i == len(objects) - 1:
                    dmm_lines.append(f"{obj})")  # Last item, close parenthesis
                else:
                    dmm_lines.append(f"{obj},")  # Add comma for non-last items

        # Write grid
        dmm_lines.append(f'(1,1,{z_level}) = {{"')
        for row in grid:
            dmm_lines.append("".join(row))
        dmm_lines.append('"}')

        return "\n".join(dmm_lines)

    def postprocess_map(self, map_tensor):
        """Clean up generated map to ensure valid SS13 structure"""
        layers, height, width = map_tensor.shape

        # Ensure layer 0 has valid turfs
        for y in range(height):
            for x in range(width):
                turf_id = int(map_tensor[0, y, x])
                turf_token = self.tokenizer.id_to_token.get(turf_id, "")

                # If no turf or invalid turf, set to plating
                if not turf_token.startswith("/turf/"):
                    # Find the ID for basic plating
                    plating_id = self.tokenizer.token_to_id.get(
                        "/turf/open/floor/plating", self.tokenizer.UNK_ID
                    )
                    map_tensor[0, y, x] = plating_id

        return map_tensor

    def save_maps(self, maps, output_dir="generated_maps"):
        """Save generated maps as DMM files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        print(f"\nSaving {len(maps)} maps to {output_dir}...")

        for i, map_tensor in enumerate(tqdm(maps, desc="Saving maps")):
            dmm_content = self.tensor_to_dmm(map_tensor, map_name=f"generated_{i}")

            file_path = output_path / f"generated_map_{i:03d}.dmm"
            with open(file_path, "w") as f:
                f.write(dmm_content)

        print(f"Maps saved successfully!")

    def visualize_layers(self, map_tensor):
        """Print a text visualization of each layer"""
        layers, height, width = map_tensor.shape

        print("\n=== Layer Visualization ===")
        for layer in range(layers):
            # Check if layer has any non-empty tokens
            layer_data = map_tensor[layer]
            unique_tokens = np.unique(layer_data)

            if len(unique_tokens) == 1 and unique_tokens[0] == self.tokenizer.EMPTY_ID:
                continue  # Skip empty layers

            print(f"\n--- Layer {layer} ---")

            # Create character map for visualization
            token_to_char = {}
            char_index = 0
            chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

            for token_id in unique_tokens:
                if token_id != self.tokenizer.EMPTY_ID:
                    if char_index < len(chars):
                        token_to_char[token_id] = chars[char_index]
                    else:
                        token_to_char[token_id] = "?"
                    char_index += 1
                else:
                    token_to_char[token_id] = "."

            # Print grid
            for y in range(height):
                row = ""
                for x in range(width):
                    token_id = int(layer_data[y, x])
                    row += token_to_char.get(token_id, "?")
                print(row)

            # Print legend
            print("\nLegend:")
            for token_id, char in token_to_char.items():
                if token_id != self.tokenizer.EMPTY_ID:
                    token = self.tokenizer.id_to_token.get(
                        token_id, f"<UNK:{token_id}>"
                    )
                    print(f"  {char}: {token}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate SS13 maps using trained diffusion model"
    )
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--tokenizer", type=str, default="tiles.json", help="Path to tokenizer JSON"
    )
    parser.add_argument(
        "--num-maps", type=int, default=4, help="Number of maps to generate"
    )
    parser.add_argument("--width", type=int, default=16, help="Map width")
    parser.add_argument("--height", type=int, default=16, help="Map height")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--output-dir", type=str, default="generated_maps", help="Output directory"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize generated maps"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Initialize generator
    generator = SS13MapGenerator(args.checkpoint, args.tokenizer)

    # Generate maps
    maps = generator.generate_maps(
        num_maps=args.num_maps,
        width=args.width,
        height=args.height,
        temperature=args.temperature,
    )

    # Visualize if requested
    if args.visualize:
        for i, map_tensor in enumerate(maps[:2]):  # Visualize first 2 maps
            print(f"\n{'='*50}")
            print(f"Map {i}")
            print("=" * 50)
            generator.visualize_layers(map_tensor)

    # Save maps
    generator.save_maps(maps, args.output_dir)

    # Show example of first generated map
    print("\n=== Example Generated DMM ===")
    print(
        generator.tensor_to_dmm(maps[0])[:1000] + "..."
        if len(maps) > 0
        else "No maps generated"
    )


if __name__ == "__main__":
    main()

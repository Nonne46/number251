import argparse
import string
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
        print(f"Timesteps: {self.model.timesteps}")

    @torch.no_grad()
    def generate_maps(
        self,
        num_maps=1,
        width=16,
        height=16,
        temperature=1.0,
        show_progress=True,
        method="basic",
        guidance_steps=5,
    ):
        """Generate maps using the diffusion model"""
        print(f"\nGenerating {num_maps} maps of size {width}x{height}...")
        print(f"Method: {method}, Temperature: {temperature}")

        batch_size = min(num_maps, 4)  # Generate in batches of 4
        all_maps = []

        for batch_start in range(0, num_maps, batch_size):
            current_batch_size = min(batch_size, num_maps - batch_start)

            # Generate batch
            if method == "guided":
                maps = self._sample_batch_guided(
                    current_batch_size,
                    width,
                    height,
                    temperature,
                    guidance_steps,
                    show_progress,
                )
            else:
                maps = self._sample_batch_basic(
                    current_batch_size, width, height, temperature, show_progress
                )
            all_maps.extend(maps)

        return all_maps

    def _sample_batch_basic(
        self, batch_size, width, height, temperature=1.0, show_progress=True
    ):
        """Sample using the improved basic method from the model"""
        shape = (batch_size, self.model.hparams.layers, height, width)
        maps_tensor = self.model.sample(shape, self.device, temperature)

        # Convert to numpy
        maps_np = maps_tensor.cpu().numpy()
        return [maps_np[i] for i in range(batch_size)]

    def _sample_batch_guided(
        self,
        batch_size,
        width,
        height,
        temperature=1.0,
        guidance_steps=5,
        show_progress=True,
    ):
        """Sample using guided method for better quality"""
        shape = (batch_size, self.model.hparams.layers, height, width)
        maps_tensor = self.model.sample_with_guidance(
            shape, self.device, temperature, guidance_steps
        )

        # Convert to numpy
        maps_np = maps_tensor.cpu().numpy()
        return [maps_np[i] for i in range(batch_size)]

    def _sample_batch_legacy(
        self, batch_size, width, height, temperature=1.0, show_progress=True
    ):
        """Legacy sampling method (kept for compatibility)"""
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

        # Denoise step by step using the old method
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

            # For legacy compatibility, just use the prediction
            x = x_pred

            # Show progress
            unique_tokens = len(torch.unique(x))
            pbar.set_postfix({"t": t, "unique_tokens": unique_tokens})

        # Convert to numpy
        maps_np = x.cpu().numpy()
        return [maps_np[i] for i in range(batch_size)]

    @torch.no_grad()
    def test_reconstruction(self, test_map, noise_level=0.3):
        """Test if model can reconstruct a given map with some noise"""
        print(f"\nTesting reconstruction with noise level {noise_level}...")

        # Convert to tensor if needed
        if isinstance(test_map, np.ndarray):
            test_map = torch.tensor(test_map, dtype=torch.long, device=self.device)

        if test_map.dim() == 3:
            test_map = test_map.unsqueeze(0)  # Add batch dimension

        batch_size = test_map.shape[0]

        # Add noise to the map
        t = torch.full(
            (batch_size,), int(self.model.timesteps * noise_level), device=self.device
        )
        x_noisy, mask = self.model.q_sample(test_map, t)

        print(f"Masked {mask.float().mean().item():.2%} of tokens")

        # Try to reconstruct
        shape = test_map.shape
        x_reconstructed = self.model.sample(shape, self.device, temperature=0.8)

        # Calculate similarity
        correct = (x_reconstructed == test_map).float().mean().item()
        print(f"Reconstruction accuracy: {correct:.2%}")

        return x_reconstructed.cpu().numpy()

    def analyze_generation_quality(self, maps, sample_size=None):
        """Analyze the quality and diversity of generated maps"""
        if sample_size and len(maps) > sample_size:
            maps = maps[:sample_size]

        print(f"\n=== Generation Quality Analysis ===")
        print(f"Analyzing {len(maps)} maps...")

        # Convert to tensors for analysis
        maps_tensor = torch.stack([torch.tensor(m) for m in maps])

        # Basic statistics
        total_positions = (
            maps_tensor.shape[0] * maps_tensor.shape[2] * maps_tensor.shape[3]
        )

        # Layer usage statistics
        layer_usage = {}
        for layer in range(maps_tensor.shape[1]):
            layer_data = maps_tensor[:, layer]
            non_empty = (layer_data != self.tokenizer.EMPTY_ID).float().mean().item()
            unique_tokens = len(torch.unique(layer_data))
            layer_usage[layer] = {
                "occupancy": non_empty,
                "unique_tokens": unique_tokens,
            }

        print("\nLayer Usage:")
        for layer, stats in layer_usage.items():
            layer_type = "Turf" if layer == 0 else f"Object {layer}"
            print(
                f"  Layer {layer} ({layer_type}): {stats['occupancy']:.1%} occupancy, "
                f"{stats['unique_tokens']} unique tokens"
            )

        # Token frequency analysis
        all_tokens = maps_tensor.flatten()
        unique_tokens, counts = torch.unique(all_tokens, return_counts=True)

        print(f"\nToken Diversity:")
        print(f"  Total unique tokens used: {len(unique_tokens)}")
        print(
            f"  Vocabulary utilization: {len(unique_tokens)}/{self.model.vocab_size} "
            f"({len(unique_tokens)/self.model.vocab_size:.1%})"
        )

        # Most common tokens
        top_indices = torch.argsort(counts, descending=True)[:10]
        print(f"\nMost common tokens:")
        for i, idx in enumerate(top_indices):
            token_id = unique_tokens[idx].item()
            count = counts[idx].item()
            token_name = self.tokenizer.id_to_token.get(token_id, f"<UNK:{token_id}>")
            percentage = count / len(all_tokens) * 100
            print(f"  {i+1:2d}. {token_name:<30} {count:6d} ({percentage:5.1f}%)")

        # Map diversity (how similar are maps to each other)
        if len(maps) > 1:
            similarities = []
            for i in range(min(10, len(maps))):
                for j in range(i + 1, min(10, len(maps))):
                    similarity = (
                        (maps_tensor[i] == maps_tensor[j]).float().mean().item()
                    )
                    similarities.append(similarity)

            avg_similarity = np.mean(similarities)
            print(f"\nMap Diversity:")
            print(f"  Average pairwise similarity: {avg_similarity:.1%}")
            print(f"  Diversity score: {1-avg_similarity:.1%}")

        return layer_usage, unique_tokens, counts

    def tensor_to_dmm(self, tensor_map, map_name="generated", z_level=1):
        layers, height, width = tensor_map.shape

        # Dictionary to store unique tile combinations
        tile_combinations = {}
        combination_to_letter = {}

        # Grid to store letter codes
        grid = [[None for _ in range(width)] for _ in range(height)]

        # Technical tokens to ignore
        tech_tokens = list(self.tokenizer.reserved_tokens.keys())

        # First pass: collect all unique combinations (excluding empty positions)
        all_combinations = []
        for y in range(height):
            for x in range(width):
                # Get all objects at this position across all layers
                objects_at_pos = []

                for layer in range(layers):
                    token_id = int(tensor_map[layer, y, x])

                    # Get token string
                    token = self.tokenizer.id_to_token.get(
                        token_id, f"<UNK:{token_id}>"
                    )

                    # Skip technical tokens
                    if token in tech_tokens or token.startswith("<UNK:"):
                        continue

                    objects_at_pos.append(token)

                # Only process non-empty positions
                if objects_at_pos:
                    # Create tuple for unique combination
                    combination = tuple(objects_at_pos)
                    if combination not in all_combinations:
                        all_combinations.append(combination)

        # Generate keys for all combinations (excluding empty)
        num_combinations = len(all_combinations)

        # Available characters for keys
        chars = list(string.ascii_lowercase + string.ascii_uppercase)

        # Determine key length needed
        if num_combinations <= len(chars):
            key_length = 1
        elif num_combinations <= len(chars) ** 2:
            key_length = 2
        elif num_combinations <= len(chars) ** 3:
            key_length = 3
        else:
            key_length = 4  # Should be enough for most cases

        # Generate keys
        for i, combination in enumerate(all_combinations):
            if key_length == 1:
                key = chars[i]
            else:
                # Generate multi-character key
                key = ""
                temp_i = i
                for _ in range(key_length):
                    key = chars[temp_i % len(chars)] + key
                    temp_i //= len(chars)

            combination_to_letter[combination] = key
            tile_combinations[key] = combination

        # Second pass: build the grid
        for y in range(height):
            for x in range(width):
                # Recreate the combination for this position
                objects_at_pos = []

                for layer in range(layers):
                    token_id = int(tensor_map[layer, y, x])

                    token = self.tokenizer.id_to_token.get(
                        token_id, f"<UNK:{token_id}>"
                    )

                    if token in tech_tokens or token.startswith("<UNK:"):
                        continue

                    objects_at_pos.append(token)

                # If completely empty, use spaces equal to key length
                if not objects_at_pos:
                    grid[y][x] = " " * key_length
                else:
                    combination = tuple(objects_at_pos)
                    grid[y][x] = combination_to_letter[combination]

        # Build DMM string
        dmm_lines = []

        # Write tile definitions - sorted for consistency (only non-empty combinations)
        for key in sorted(tile_combinations.keys()):
            objects = tile_combinations[key]
            dmm_lines.append(f'"{key}" = (')

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
        "--method",
        type=str,
        default="basic",
        choices=["basic", "guided", "legacy"],
        help="Sampling method (basic=improved, guided=high quality, legacy=old method)",
    )
    parser.add_argument(
        "--guidance-steps",
        type=int,
        default=5,
        help="Number of guidance steps for guided sampling",
    )
    parser.add_argument(
        "--output-dir", type=str, default="generated_maps", help="Output directory"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Visualize generated maps"
    )
    parser.add_argument(
        "--analyze", action="store_true", help="Analyze generation quality"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Set random seed to {args.seed}")

    # Initialize generator
    generator = SS13MapGenerator(args.checkpoint, args.tokenizer)

    # Generate maps
    maps = generator.generate_maps(
        num_maps=args.num_maps,
        width=args.width,
        height=args.height,
        temperature=args.temperature,
        method=args.method,
        guidance_steps=args.guidance_steps,
    )

    # Analyze quality if requested
    if args.analyze:
        generator.analyze_generation_quality(maps)

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

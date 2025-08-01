import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from inference import SS13MapGenerator


class ContinuousMapGenerator:
    def __init__(self, checkpoint_path, tokenizer_path, chunk_size=16):
        """Initialize continuous map generator"""
        self.generator = SS13MapGenerator(checkpoint_path, tokenizer_path)
        self.chunk_size = chunk_size
        self.overlap = chunk_size // 2  # 50% overlap
        self.device = self.generator.device

    def generate_spiral_positions(self, map_width, map_height):
        """Generate positions for chunks in spiral pattern"""
        stride = self.overlap
        positions = []

        # Calculate how many chunks we need in each direction
        chunks_x = max(1, (map_width - self.chunk_size) // stride + 1)
        chunks_y = max(1, (map_height - self.chunk_size) // stride + 1)

        # Add final chunks to cover edges
        if map_width > self.chunk_size:
            chunks_x += 1
        if map_height > self.chunk_size:
            chunks_y += 1

        # Generate all positions first
        all_positions = []
        for y in range(chunks_y):
            for x in range(chunks_x):
                pos_x = min(x * stride, map_width - self.chunk_size)
                pos_y = min(y * stride, map_height - self.chunk_size)
                all_positions.append((pos_x, pos_y))

        # Convert to spiral order
        visited = set()

        # Start from center
        center_x, center_y = chunks_x // 2, chunks_y // 2

        # Spiral directions: right, down, left, up
        dx = [1, 0, -1, 0]
        dy = [0, 1, 0, -1]

        x, y = center_x, center_y
        direction = 0
        steps = 1

        # Add center position first
        if 0 <= x < chunks_x and 0 <= y < chunks_y:
            pos_x = min(x * stride, map_width - self.chunk_size)
            pos_y = min(y * stride, map_height - self.chunk_size)
            positions.append((pos_x, pos_y))
            visited.add((x, y))

        # Spiral outward
        while len(visited) < len(all_positions):
            for _ in range(2):  # Each step count is used twice
                for _ in range(steps):
                    x += dx[direction]
                    y += dy[direction]

                    if (
                        0 <= x < chunks_x
                        and 0 <= y < chunks_y
                        and (x, y) not in visited
                    ):
                        pos_x = min(x * stride, map_width - self.chunk_size)
                        pos_y = min(y * stride, map_height - self.chunk_size)
                        positions.append((pos_x, pos_y))
                        visited.add((x, y))

                direction = (direction + 1) % 4
            steps += 1

            # Safety break
            if steps > max(chunks_x, chunks_y) * 2:
                break

        # Add any remaining positions (shouldn't happen but safety first)
        for y in range(chunks_y):
            for x in range(chunks_x):
                if (x, y) not in visited:
                    pos_x = min(x * stride, map_width - self.chunk_size)
                    pos_y = min(y * stride, map_height - self.chunk_size)
                    positions.append((pos_x, pos_y))

        return positions

    def create_overlap_mask(self, full_map, pos_x, pos_y):
        """Create mask to preserve existing content in overlap regions"""
        mask = torch.zeros(
            (self.generator.model.hparams.layers, self.chunk_size, self.chunk_size),
            dtype=torch.bool,
            device=self.device,
        )

        # Check if we have existing content to preserve
        chunk = full_map[
            :, pos_y : pos_y + self.chunk_size, pos_x : pos_x + self.chunk_size
        ]

        # Mark positions that already have content (not empty or pad tokens)
        existing_content = (chunk != self.generator.tokenizer.EMPTY_ID) & (
            chunk != self.generator.tokenizer.PAD_ID
        )
        mask = existing_content

        return mask

    @torch.no_grad()
    def sample_with_mask(
        self, shape, existing_content=None, mask=None, temperature=1.0, method="basic"
    ):
        """Generate chunk with masking to preserve existing content"""
        batch_size, layers, h, w = shape

        if existing_content is None or mask is None:
            # No masking, generate normally
            if method == "guided":
                return self.generator._sample_batch_guided(1, w, h, temperature)[0]
            else:
                return self.generator._sample_batch_basic(1, w, h, temperature)[0]

        # Start with existing content
        x = (
            existing_content.clone().unsqueeze(0)
            if existing_content.dim() == 3
            else existing_content.clone()
        )

        # Add mask dimension if needed
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)

        # Iterative demasking with preservation
        for t in reversed(range(self.generator.model.timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device)

            # Predict logits
            logits = self.generator.model(x, t_batch)

            # Apply temperature
            logits = logits / temperature

            # Get probabilities
            probs = torch.softmax(logits, dim=2)

            # Sample new tokens
            probs_flat = probs.permute(0, 1, 3, 4, 2).reshape(
                -1, self.generator.model.vocab_size
            )
            sampled_tokens = torch.multinomial(probs_flat, num_samples=1)
            sampled_tokens = sampled_tokens.reshape(batch_size, layers, h, w)

            if t > 0:
                # Update probability - more conservative for masked generation
                update_prob = max(
                    0.05, self.generator.model.mask_probs[t - 1].item() * 0.2
                )
                update_mask = (
                    torch.rand(batch_size, layers, h, w, device=self.device)
                    < update_prob
                )

                # Don't update masked positions (where mask is True)
                update_mask = update_mask & ~mask

                x = torch.where(update_mask, sampled_tokens, x)
            else:
                # Final step - don't update masked positions
                final_mask = ~mask
                x = torch.where(final_mask, sampled_tokens, x)

        return x.squeeze(0).cpu().numpy() if x.dim() == 4 else x.cpu().numpy()

    @torch.no_grad()
    def sample_with_mask_guided(
        self, shape, existing_content=None, mask=None, temperature=1.0, guidance_steps=5
    ):
        """Enhanced masked sampling with self-guidance"""
        batch_size, layers, h, w = shape

        if existing_content is None or mask is None:
            return self.generator._sample_batch_guided(1, w, h, temperature)[0]

        # Start with existing content
        x = (
            existing_content.clone().unsqueeze(0)
            if existing_content.dim() == 3
            else existing_content.clone()
        )

        if mask.dim() == 3:
            mask = mask.unsqueeze(0)

        for t in reversed(range(self.generator.model.timesteps)):
            t_batch = torch.full((batch_size,), t, device=self.device)

            # Multiple guidance steps at each timestep
            steps = guidance_steps if t > self.generator.model.timesteps // 2 else 1
            for _ in range(steps):
                logits = self.generator.model(x, t_batch)
                logits = logits / temperature
                probs = torch.softmax(logits, dim=2)

                # Sample with higher confidence
                probs_flat = probs.permute(0, 1, 3, 4, 2).reshape(
                    -1, self.generator.model.vocab_size
                )
                sampled_tokens = torch.multinomial(probs_flat, num_samples=1)
                sampled_tokens = sampled_tokens.reshape(batch_size, layers, h, w)

                if t > 0:
                    # Only update most uncertain positions, but not masked ones
                    confidence = torch.max(probs, dim=2)[0]
                    uncertainty_threshold = torch.quantile(confidence.flatten(), 0.3)
                    update_mask = (confidence < uncertainty_threshold) & ~mask
                    x = torch.where(update_mask, sampled_tokens, x)
                else:
                    # Final step - preserve masked content
                    final_mask = ~mask
                    x = torch.where(final_mask, sampled_tokens, x)

        return x.squeeze(0).cpu().numpy() if x.dim() == 4 else x.cpu().numpy()

    def generate_large_map(self, width, height, temperature=1.0, method="basic"):
        """Generate large map using spiral pattern with overlap"""
        print(f"Generating {width}x{height} map using spiral pattern...")
        print(f"Chunk size: {self.chunk_size}, Overlap: {self.overlap}")

        # Initialize full map with empty tokens
        layers = self.generator.model.hparams.layers
        full_map = torch.full(
            (layers, height, width),
            self.generator.tokenizer.EMPTY_ID,
            dtype=torch.long,
            device=self.device,
        )

        # Get spiral positions
        positions = self.generate_spiral_positions(width, height)
        print(f"Will generate {len(positions)} chunks")

        # Generate chunks in spiral order
        for i, (pos_x, pos_y) in enumerate(tqdm(positions, desc="Generating chunks")):
            # Get existing content in this region
            existing_chunk = full_map[
                :, pos_y : pos_y + self.chunk_size, pos_x : pos_x + self.chunk_size
            ]

            # Create mask for overlap regions (True = preserve, False = generate)
            mask = self.create_overlap_mask(full_map, pos_x, pos_y)

            # Generate chunk with masking
            shape = (1, layers, self.chunk_size, self.chunk_size)

            if method == "guided":
                chunk = self.sample_with_mask_guided(
                    shape, existing_chunk, mask, temperature
                )
            else:
                chunk = self.sample_with_mask(
                    shape, existing_chunk, mask, temperature, method="basic"
                )

            # Convert to tensor if needed
            if isinstance(chunk, np.ndarray):
                chunk_tensor = torch.tensor(chunk, dtype=torch.long, device=self.device)
            else:
                chunk_tensor = chunk

            # Place chunk in full map
            full_map[
                :, pos_y : pos_y + self.chunk_size, pos_x : pos_x + self.chunk_size
            ] = chunk_tensor

        return full_map.cpu().numpy()

    def refine_map(self, map_tensor, noise_level=0.3, temperature=0.8, chunk_overlap=8):
        """Refine existing map by re-processing chunks with controlled noise"""
        print(f"Refining map with noise level {noise_level}...")

        if isinstance(map_tensor, np.ndarray):
            map_tensor = torch.tensor(map_tensor, dtype=torch.long, device=self.device)

        layers, height, width = map_tensor.shape
        stride = self.chunk_size - chunk_overlap

        # Generate grid positions (not spiral, just systematic)
        positions = []
        y = 0
        while y + self.chunk_size <= height:
            x = 0
            while x + self.chunk_size <= width:
                positions.append((x, y))
                x += stride
            # Add right edge chunk if needed
            if width > self.chunk_size and x < width:
                positions.append((width - self.chunk_size, y))
            y += stride

        # Add bottom edge chunks if needed
        if height > self.chunk_size and y < height:
            x = 0
            while x + self.chunk_size <= width:
                positions.append((x, height - self.chunk_size))
                x += stride
            # Bottom-right corner
            if width > self.chunk_size and x < width:
                positions.append((width - self.chunk_size, height - self.chunk_size))

        print(f"Refining {len(positions)} chunks")

        # Process each chunk
        for pos_x, pos_y in tqdm(positions, desc="Refining chunks"):
            # Extract chunk
            chunk = map_tensor[
                :, pos_y : pos_y + self.chunk_size, pos_x : pos_x + self.chunk_size
            ].clone()

            # Create noise mask - randomly select positions to refine
            noise_mask = torch.rand(chunk.shape, device=self.device) < noise_level

            # Create preservation mask - preserve some original content
            preserve_mask = (
                torch.rand(chunk.shape, device=self.device) < 0.4
            )  # Keep 40% unchanged

            # Combine masks - preserve some content, noise the rest
            final_mask = preserve_mask & ~noise_mask

            # Generate refined chunk with masking
            shape = (1, layers, self.chunk_size, self.chunk_size)
            refined_chunk = self.sample_with_mask(
                shape, chunk, final_mask, temperature, method="basic"
            )

            # Convert to tensor if needed
            if isinstance(refined_chunk, np.ndarray):
                refined_tensor = torch.tensor(
                    refined_chunk, dtype=torch.long, device=self.device
                )
            else:
                refined_tensor = refined_chunk

            # Place back in full map
            map_tensor[
                :, pos_y : pos_y + self.chunk_size, pos_x : pos_x + self.chunk_size
            ] = refined_tensor

        return map_tensor.cpu().numpy()

    def generate_grid_positions(self, width, height):
        """Generate positions for systematic grid (used in refinement)"""
        stride = self.overlap
        positions = []

        y = 0
        while y + self.chunk_size <= height:
            x = 0
            while x + self.chunk_size <= width:
                positions.append((x, y))
                x += stride
            y += stride

        return positions


def main():
    parser = argparse.ArgumentParser(
        description="Generate large SS13 maps continuously"
    )
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--tokenizer", default="tiles.json", help="Path to tokenizer")
    parser.add_argument("--width", type=int, default=64, help="Target map width")
    parser.add_argument("--height", type=int, default=64, help="Target map height")
    parser.add_argument(
        "--chunk-size", type=int, default=16, help="Size of generation chunks"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--method",
        choices=["basic", "guided"],
        default="basic",
        help="Generation method",
    )
    parser.add_argument("--output-dir", default="large_maps", help="Output directory")
    parser.add_argument("--refine", action="store_true", help="Refine generated map")
    parser.add_argument(
        "--refine-noise", type=float, default=0.3, help="Noise level for refinement"
    )
    parser.add_argument(
        "--refine-temp", type=float, default=0.8, help="Temperature for refinement"
    )
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument(
        "--num-maps", type=int, default=1, help="Number of maps to generate"
    )

    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        print(f"Set random seed to {args.seed}")

    # Initialize generator
    generator = ContinuousMapGenerator(args.checkpoint, args.tokenizer, args.chunk_size)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    for map_idx in range(args.num_maps):
        print(f"\n=== Generating Map {map_idx + 1}/{args.num_maps} ===")

        # Generate large map
        large_map = generator.generate_large_map(
            args.width, args.height, temperature=args.temperature, method=args.method
        )

        # Refine if requested
        if args.refine:
            print("\nRefining map...")
            large_map = generator.refine_map(
                large_map, noise_level=args.refine_noise, temperature=args.refine_temp
            )

        # Convert to DMM and save
        dmm_content = generator.generator.tensor_to_dmm(
            large_map, f"large_map_{map_idx}"
        )

        output_file = (
            output_path / f"large_map_{map_idx:03d}_{args.width}x{args.height}.dmm"
        )
        with open(output_file, "w") as f:
            f.write(dmm_content)

        print(f"Saved map to {output_file}")

        # Show some stats
        unique_tokens = len(np.unique(large_map))
        occupancy = (large_map != generator.generator.tokenizer.EMPTY_ID).mean()
        print(f"Map stats: {unique_tokens} unique tokens, {occupancy:.1%} occupancy")

    print(f"\nAll maps saved to {output_path}")


if __name__ == "__main__":
    main()

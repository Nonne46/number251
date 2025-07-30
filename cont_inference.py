import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from inference import SS13MapGenerator


class SS13ContinuousGenerator(SS13MapGenerator):
    def __init__(self, checkpoint_path, tokenizer_path):
        super().__init__(checkpoint_path, tokenizer_path)

    def create_generation_mask(self, shape, region):
        """Create a mask for the region to generate
        region: (y_start, y_end, x_start, x_end)
        Returns mask where True = generate, False = keep original
        """
        layers, height, width = shape
        mask = np.zeros((layers, height, width), dtype=bool)
        y_start, y_end, x_start, x_end = region
        mask[:, y_start:y_end, x_start:x_end] = True
        return mask

    def blend_overlap(self, existing, generated, region, overlap=2):
        """Blend generated content with existing content in overlap region"""
        y_start, y_end, x_start, x_end = region
        result = existing.copy()

        # Direct copy for non-overlap region
        result[
            :, y_start + overlap : y_end - overlap, x_start + overlap : x_end - overlap
        ] = generated[
            :, y_start + overlap : y_end - overlap, x_start + overlap : x_end - overlap
        ]

        # Blend overlap regions with preference for existing content
        for dy in range(overlap):
            for dx in range(overlap):
                # Top overlap
                if y_start + dy < y_end:
                    blend_factor = (dy + 1) / (overlap + 1)
                    mask = (
                        np.random.random((existing.shape[0], x_end - x_start))
                        < blend_factor
                    )
                    result[:, y_start + dy, x_start:x_end][mask] = generated[
                        :, y_start + dy, x_start:x_end
                    ][mask]

                # Left overlap
                if x_start + dx < x_end:
                    blend_factor = (dx + 1) / (overlap + 1)
                    mask = (
                        np.random.random((existing.shape[0], y_end - y_start))
                        < blend_factor
                    )
                    result[:, y_start:y_end, x_start + dx][mask] = generated[
                        :, y_start:y_end, x_start + dx
                    ][mask]

        return result

    @torch.no_grad()
    def generate_continuous(
        self,
        target_width,
        target_height,
        chunk_size=16,
        overlap=4,
        temperature=1.0,
        initial_map=None,
    ):
        """Generate a large map by generating overlapping chunks"""

        print(
            f"\nGenerating {target_width}x{target_height} map using {chunk_size}x{chunk_size} chunks..."
        )
        print(f"Overlap: {overlap} tiles")

        layers = self.model.hparams.layers

        # Initialize the full map
        if initial_map is not None:
            # Start from provided map
            full_map = initial_map
            current_h, current_w = initial_map.shape[1:3]
        else:
            # Start with empty map (will generate first chunk)
            full_map = np.full(
                (layers, target_height, target_width),
                self.tokenizer.EMPTY_ID,
                dtype=np.int64,
            )
            current_h = 0
            current_w = 0

        # Calculate chunks needed
        stride = chunk_size - overlap
        chunks_x = max(1, (target_width - current_w + stride - 1) // stride)
        chunks_y = max(1, (target_height - current_h + stride - 1) // stride)

        total_chunks = chunks_x * chunks_y
        chunk_count = 0

        # Generate initial chunk if starting from scratch
        if initial_map is None:
            print("\nGenerating initial chunk...")
            initial = self._sample_batch_basic(
                1, chunk_size, chunk_size, temperature, show_progress=True
            )[0]
            full_map[:, :chunk_size, :chunk_size] = initial
            current_h = chunk_size
            current_w = chunk_size
            chunk_count = 1

        # Progress bar for overall generation
        pbar = tqdm(total=total_chunks - chunk_count, desc="Generating chunks")

        # Generate chunks in a spiral pattern from center outward
        # This gives better coherence than left-to-right generation
        for radius in range(1, max(chunks_x, chunks_y)):
            for direction in ["right", "down", "left", "up"]:
                if direction == "right" and current_w < target_width:
                    # Generate to the right
                    for _ in range(
                        min(radius, (target_width - current_w + stride - 1) // stride)
                    ):
                        if current_w >= target_width:
                            break

                        y_start = max(0, current_h - chunk_size)
                        y_end = min(target_height, y_start + chunk_size)
                        x_start = max(0, current_w - overlap)
                        x_end = min(target_width, x_start + chunk_size)

                        generated = self._generate_chunk_with_context(
                            full_map,
                            (y_start, y_end, x_start, x_end),
                            temperature,
                            chunk_size,
                        )

                        full_map = self.blend_overlap(
                            full_map,
                            generated,
                            (y_start, y_end, x_start, x_end),
                            overlap,
                        )

                        current_w = x_end
                        chunk_count += 1
                        pbar.update(1)

                elif direction == "down" and current_h < target_height:
                    # Generate downward
                    for _ in range(
                        min(radius, (target_height - current_h + stride - 1) // stride)
                    ):
                        if current_h >= target_height:
                            break

                        y_start = max(0, current_h - overlap)
                        y_end = min(target_height, y_start + chunk_size)
                        x_start = max(0, current_w - chunk_size)
                        x_end = min(target_width, x_start + chunk_size)

                        generated = self._generate_chunk_with_context(
                            full_map,
                            (y_start, y_end, x_start, x_end),
                            temperature,
                            chunk_size,
                        )

                        full_map = self.blend_overlap(
                            full_map,
                            generated,
                            (y_start, y_end, x_start, x_end),
                            overlap,
                        )

                        current_h = y_end
                        chunk_count += 1
                        pbar.update(1)

        pbar.close()
        return full_map

    def _generate_chunk_with_context(self, full_map, region, temperature, chunk_size):
        """Generate a chunk considering the context from full_map"""
        y_start, y_end, x_start, x_end = region
        layers = full_map.shape[0]

        # Extract the region with context
        context_tensor = torch.zeros(
            (1, layers, chunk_size, chunk_size), dtype=torch.long, device=self.device
        )

        # Fill with existing data where available
        for i in range(chunk_size):
            for j in range(chunk_size):
                y_idx = y_start + i
                x_idx = x_start + j

                if 0 <= y_idx < full_map.shape[1] and 0 <= x_idx < full_map.shape[2]:
                    context_tensor[0, :, i, j] = torch.tensor(full_map[:, y_idx, x_idx])
                else:
                    context_tensor[0, :, i, j] = self.tokenizer.EMPTY_ID

        # Create mask for generation (True = need to generate)
        generation_mask = context_tensor[0, 0] == self.tokenizer.EMPTY_ID

        # Run masked diffusion
        generated = self._sample_with_mask(context_tensor, generation_mask, temperature)

        # Create full-size output
        result = full_map.copy()

        # Copy generated content back
        for i in range(chunk_size):
            for j in range(chunk_size):
                y_idx = y_start + i
                x_idx = x_start + j

                if 0 <= y_idx < full_map.shape[1] and 0 <= x_idx < full_map.shape[2]:
                    result[:, y_idx, x_idx] = generated[0, :, i, j].cpu().numpy()

        return result

    @torch.no_grad()
    def _sample_with_mask(self, context, generation_mask, temperature=1.0):
        """Sample using the diffusion model with masking"""
        batch_size = context.shape[0]
        layers, h, w = context.shape[1:]
        timesteps = self.model.timesteps

        # Start with context (keep existing tokens)
        x = context.clone()

        # Add noise only to areas that need generation
        noise = torch.randint(0, self.model.vocab_size, x.shape, device=self.device)
        mask_expanded = generation_mask.unsqueeze(0).expand(layers, -1, -1)
        x[0][mask_expanded] = noise[0][mask_expanded]

        # Denoise step by step
        for t in tqdm(reversed(range(timesteps)), desc="Denoising", leave=False):
            t_batch = torch.full((batch_size,), t, device=self.device)

            # Predict denoised tokens
            logits = self.model(x, t_batch)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            probs = torch.softmax(logits, dim=2)

            # Sample from distribution
            x_pred = torch.multinomial(
                probs.permute(0, 1, 3, 4, 2).reshape(-1, self.model.vocab_size),
                num_samples=1,
            ).reshape(batch_size, layers, h, w)

            x = x_pred

        return x


def main():
    parser = argparse.ArgumentParser(
        description="Generate large SS13 maps through continuous generation"
    )
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument(
        "--tokenizer", type=str, default="tiles.json", help="Path to tokenizer JSON"
    )
    parser.add_argument("--width", type=int, default=32, help="Target map width")
    parser.add_argument("--height", type=int, default=32, help="Target map height")
    parser.add_argument(
        "--chunk-size", type=int, default=16, help="Size of generation chunks"
    )
    parser.add_argument("--overlap", type=int, default=4, help="Overlap between chunks")
    parser.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature"
    )
    parser.add_argument(
        "--output-dir", type=str, default="continuous_maps", help="Output directory"
    )
    parser.add_argument(
        "--num-maps", type=int, default=1, help="Number of maps to generate"
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--initial-map",
        type=str,
        default=None,
        help="Path to initial map chunk to extend",
    )

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Initialize generator
    generator = SS13ContinuousGenerator(args.checkpoint, args.tokenizer)

    # Load initial map if provided
    initial_map = None
    if args.initial_map:
        print(f"Loading initial map from {args.initial_map}...")
        # This would need implementation to load a DMM file back to tensor
        # For now, we'll start from scratch
        pass

    # Generate maps
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True)

    for i in range(args.num_maps):
        print(f"\n{'='*60}")
        print(f"Generating map {i+1}/{args.num_maps}")
        print("=" * 60)

        # Generate large map
        large_map = generator.generate_continuous(
            target_width=args.width,
            target_height=args.height,
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            temperature=args.temperature,
            initial_map=initial_map,
        )

        # Convert to DMM
        print("\nConverting to DMM format...")
        dmm_content = generator.tensor_to_dmm(large_map, map_name=f"continuous_{i}")

        # Save
        file_path = (
            output_path / f"continuous_map_{i:03d}_{args.width}x{args.height}.dmm"
        )
        with open(file_path, "w") as f:
            f.write(dmm_content)

        print(f"Saved to {file_path}")

        # Show stats
        unique_tokens = len(np.unique(large_map))
        print(f"Map statistics:")
        print(f"  - Size: {args.width}x{args.height}")
        print(f"  - Unique tokens: {unique_tokens}")
        print(
            f"  - Non-empty tiles: {np.sum(large_map != generator.tokenizer.EMPTY_ID)}"
        )


if __name__ == "__main__":
    main()

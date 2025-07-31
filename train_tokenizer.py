import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set


class SS13TokenizerTrainer:
    def __init__(self):
        self.object_counts = Counter()
        self.total_tiles = 0
        self.total_maps = 0
        self.object_categories = defaultdict(list)

        # Reserved tokens
        self.RESERVED_TOKENS = {
            "<EMPTY>": 0,  # Empty space
            "<UNK>": 1,  # Unknown/rare objects
            "<PAD>": 2,  # Padding token
            "<MASK>": 3,  # For masked language modeling
        }

    def extract_objects_from_map(self, map_content: str) -> Set[str]:
        """Extract all unique objects from a single map file"""
        objects = set()
        tile_definitions = self.parse_tile_definitions(map_content)

        for tile_objects in tile_definitions.values():
            objects.update(tile_objects)

        return objects

    def categorize_object(self, obj_path: str) -> str:
        """Categorize object by its path for better organization"""
        if obj_path.startswith("/turf/"):
            return "turfs"
        elif obj_path.startswith("/obj/structure/"):
            return "structures"
        elif obj_path.startswith("/obj/item/"):
            return "items"
        elif obj_path.startswith("/obj/machinery/"):
            return "machinery"
        elif obj_path.startswith("/obj/effect/"):
            return "effects"
        elif obj_path.startswith("/area/"):
            return "areas"
        elif obj_path.startswith("/obj/"):
            return "objects"
        else:
            return "other"

    def _detect_tile_id_length(self, map_content: str) -> int:
        """Detect tile identifier length by finding the first tile definition"""
        # Look for the first tile definition pattern
        match = re.search(r'"([^"]+)"\s*=\s*\(', map_content)
        if match:
            return len(match.group(1))
        return 1  # fallback

    def count_tile_occurrences(self, map_content: str) -> Dict[str, int]:
        """Count how many times each tile type appears in the map grid"""
        tile_counts = Counter()

        # Detect tile ID length first
        tile_length = self._detect_tile_id_length(map_content)

        lines = map_content.strip().split("\n")

        # Find grid definitions
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if re.match(r'\((\d+),(\d+),(\d+)\) = \{"', line):
                # Extract grid content
                i += 1
                grid_content = ""
                while i < len(lines):
                    grid_line = lines[i].strip()
                    if grid_line == '"}':
                        break
                    if grid_line and not grid_line.startswith("("):
                        # Remove quotes and add to content
                        clean_line = grid_line.strip('"')
                        grid_content += clean_line
                    i += 1

                # Count tiles based on detected length
                for j in range(0, len(grid_content), tile_length):
                    if j + tile_length <= len(grid_content):
                        tile_id = grid_content[j : j + tile_length]
                        if tile_id.strip():  # Only count non-empty tiles
                            tile_counts[tile_id] += 1
            i += 1

        return dict(tile_counts)

    def scan_directory(self, directory: str, file_pattern: str = "*.dmm") -> None:
        """Scan directory for map files and extract objects"""
        directory_path = Path(directory)

        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        # Find all map files
        if file_pattern.endswith(".dmm"):
            map_files = list(directory_path.rglob("*.dmm"))
        else:
            map_files = list(directory_path.rglob(file_pattern))

        print(f"Found {len(map_files)} map files in {directory}")

        for map_file in map_files:
            try:
                with open(map_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                # Count tile occurrences to weight object frequency
                tile_counts = self.count_tile_occurrences(content)

                # Parse tile definitions to map tiles to objects
                tile_to_objects = self.parse_tile_definitions(content)

                # Debug info for first few maps
                if self.total_maps < 3:
                    print(f"\nDebugging map: {map_file.name}")
                    print(f"Tile definitions found: {len(tile_to_objects)}")
                    print(f"Tile usage counts: {len(tile_counts)}")

                    # Show tile ID length
                    if tile_to_objects:
                        first_tile = next(iter(tile_to_objects.keys()))
                        print(f"Tile ID length: {len(first_tile)}")

                    # Show some tile definitions
                    for tile, objects in list(tile_to_objects.items())[:3]:
                        usage_count = tile_counts.get(tile, 0)
                        print(f"  Tile '{tile}' (used {usage_count} times): {objects}")

                # Count objects weighted by their tile usage
                for tile_char, count in tile_counts.items():
                    if tile_char in tile_to_objects:
                        for obj in tile_to_objects[tile_char]:
                            # if "/area" in obj:
                            #     continue
                            # if "/obj/effect" in obj:
                            #     continue
                            self.object_counts[obj] += count
                            self.object_categories[self.categorize_object(obj)].append(
                                obj
                            )

                self.total_tiles += sum(tile_counts.values())
                self.total_maps += 1

                if self.total_maps % 10 == 0:
                    print(
                        f"Processed {self.total_maps} maps, found {len(self.object_counts)} unique objects"
                    )

            except Exception as e:
                print(f"Error processing {map_file}: {e}")
                continue

    def parse_tile_definitions(self, map_content: str) -> Dict[str, List[str]]:
        """Parse tile definitions to map tile characters to object lists - fast but thorough approach"""
        tile_definitions = {}

        # Split into sections for faster processing
        # Find all tile definition blocks using a simple approach
        sections = re.split(r'\n(?="[^"]+"\s*=\s*\()', map_content)

        for section in sections:
            if not section.strip():
                continue

            lines = section.strip().split("\n")
            if not lines:
                continue

            # Check if this section starts with a tile definition
            first_line = lines[0].strip()
            tile_match = re.match(r'"([^"]+)"\s*=\s*\(', first_line)
            if not tile_match:
                continue

            tile_id = tile_match.group(1)
            objects = []

            # Process all lines in this section
            in_param_block = False
            brace_depth = 0

            for line in lines[1:]:  # Skip first line (tile definition)
                line = line.strip()

                # Skip empty lines and comments
                if not line or line.startswith("//"):
                    continue

                # Check for end of tile definition
                if line == ")":
                    break

                # Handle object lines
                if line.startswith("/"):
                    # Clean the line
                    obj_line = line.rstrip(",").strip()

                    # Check if this line ends the definition
                    ends_with_paren = obj_line.endswith(")")
                    if ends_with_paren:
                        obj_line = obj_line.rstrip(")")

                    # Handle parameter blocks
                    if "{" in obj_line:
                        base_obj = obj_line.split("{")[0].strip()
                        if base_obj:
                            objects.append(base_obj)

                        # Track brace depth for multi-line parameters
                        brace_depth += obj_line.count("{") - obj_line.count("}")
                        in_param_block = brace_depth > 0

                    elif not in_param_block:
                        # Simple object without parameters
                        if obj_line:
                            objects.append(obj_line)

                    # End definition if we found closing paren
                    if ends_with_paren:
                        break

                elif in_param_block:
                    # We're inside a parameter block, track braces
                    brace_depth += line.count("{") - line.count("}")
                    if brace_depth <= 0:
                        in_param_block = False
                        brace_depth = 0

                        # Check if parameter block line ends definition
                        if line.endswith("})"):
                            break

            if objects:
                tile_definitions[tile_id] = objects

        return tile_definitions

    def create_vocabulary(
        self, vocab_limit: int = 1000, min_frequency: int = 1
    ) -> Dict[str, Any]:
        """Create vocabulary with frequency-based filtering"""

        # Filter objects by minimum frequency
        filtered_objects = {
            obj: count
            for obj, count in self.object_counts.items()
            if count >= min_frequency
        }

        # Sort by frequency (most common first)
        sorted_objects = sorted(
            filtered_objects.items(), key=lambda x: x[1], reverse=True
        )

        # Reserve space for special tokens
        available_slots = vocab_limit - len(self.RESERVED_TOKENS)

        # Take top N most frequent objects
        vocab_objects = sorted_objects[:available_slots]

        # Create token mappings - only store token_to_id
        token_to_id = dict(self.RESERVED_TOKENS)  # Start with reserved tokens

        # Add objects to vocabulary
        next_id = max(self.RESERVED_TOKENS.values()) + 1
        for obj, _ in vocab_objects:  # Don't need frequency variable here
            token_to_id[obj] = next_id
            next_id += 1

        # Calculate statistics
        total_covered = sum(freq for _, freq in vocab_objects)
        coverage = total_covered / max(sum(self.object_counts.values()), 1)

        # Organize by categories
        vocab_by_category = defaultdict(list)
        for obj, obj_id in token_to_id.items():
            if obj not in self.RESERVED_TOKENS:
                category = self.categorize_object(obj)
                vocab_by_category[category].append(
                    {
                        "object": obj,
                        "id": obj_id,
                        "frequency": self.object_counts.get(obj, 0),
                    }
                )

        vocabulary = {
            "metadata": {
                "total_maps_processed": self.total_maps,
                "total_tiles_processed": self.total_tiles,
                "total_unique_objects": len(self.object_counts),
                "vocab_size": len(token_to_id),
                "vocab_limit": vocab_limit,
                "min_frequency": min_frequency,
                "coverage": coverage,
                "objects_excluded": len(self.object_counts) - len(vocab_objects),
            },
            "reserved_tokens": self.RESERVED_TOKENS,
            "token_to_id": token_to_id,
            "vocabulary_by_category": dict(vocab_by_category),
            "frequency_stats": {
                "most_common": sorted_objects[:20],
                "least_common_included": vocab_objects[-10:] if vocab_objects else [],
                "excluded_sample": (
                    sorted_objects[available_slots : available_slots + 10]
                    if len(sorted_objects) > available_slots
                    else []
                ),
            },
        }

        return vocabulary

    def save_vocabulary(self, vocabulary: Dict[str, Any], output_path: str) -> None:
        """Save vocabulary to JSON file"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(vocabulary, f, indent=2, ensure_ascii=False)

        print(f"Vocabulary saved to: {output_path}")
        print(f"Vocabulary size: {vocabulary['metadata']['vocab_size']}")
        print(f"Coverage: {vocabulary['metadata']['coverage']:.2%}")
        print(f"Objects excluded: {vocabulary['metadata']['objects_excluded']}")

    def print_statistics(self, vocabulary: Dict[str, Any]) -> None:
        """Print detailed statistics about the vocabulary"""
        meta = vocabulary["metadata"]

        print("\n=== Tokenizer Training Statistics ===")
        print(f"Maps processed: {meta['total_maps_processed']}")
        print(f"Tiles processed: {meta['total_tiles_processed']}")
        print(f"Unique objects found: {meta['total_unique_objects']}")
        print(f"Vocabulary size: {meta['vocab_size']}")
        print(f"Coverage: {meta['coverage']:.2%}")
        print(f"Objects excluded: {meta['objects_excluded']}")

        print("\n=== Objects by Category ===")
        for category, objects in vocabulary["vocabulary_by_category"].items():
            print(f"{category}: {len(objects)} objects")

        print("\n=== Most Common Objects ===")
        for obj, freq in vocabulary["frequency_stats"]["most_common"]:
            print(f"  {obj}: {freq}")

        if vocabulary["frequency_stats"]["excluded_sample"]:
            print("\n=== Sample Excluded Objects ===")
            for obj, freq in vocabulary["frequency_stats"]["excluded_sample"]:
                print(f"  {obj}: {freq}")


def main():
    parser = argparse.ArgumentParser(description="Train SS13 Map Tokenizer")
    parser.add_argument("map_directory", help="Directory containing map files")
    parser.add_argument(
        "--output",
        "-o",
        default="ss13_vocabulary.json",
        help="Output vocabulary file (default: ss13_vocabulary.json)",
    )
    parser.add_argument(
        "--vocab-limit",
        "-v",
        type=int,
        default=1000,
        help="Maximum vocabulary size (default: 1000)",
    )
    parser.add_argument(
        "--min-frequency",
        "-f",
        type=int,
        default=1,
        help="Minimum frequency for inclusion (default: 1)",
    )
    parser.add_argument(
        "--pattern",
        "-p",
        default="*.dmm",
        help="File pattern to match (default: *.dmm)",
    )

    args = parser.parse_args()

    # Initialize trainer
    trainer = SS13TokenizerTrainer()

    # Scan directory
    print(f"Scanning directory: {args.map_directory}")
    trainer.scan_directory(args.map_directory, args.pattern)

    # Create vocabulary
    print(f"Creating vocabulary with limit: {args.vocab_limit}")
    vocabulary = trainer.create_vocabulary(
        vocab_limit=args.vocab_limit, min_frequency=args.min_frequency
    )

    # Print statistics
    trainer.print_statistics(vocabulary)

    # Save vocabulary
    trainer.save_vocabulary(vocabulary, args.output)

    print("\nTokenizer training complete!")
    print(f"Use the vocabulary file '{args.output}' for map tokenization.")


if __name__ == "__main__":
    main()

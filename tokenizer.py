import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class MapTensor:
    """Container for tokenized map data"""

    data: np.ndarray  # Shape: [layers, height, width]
    mask: np.ndarray  # Shape: [layers, height, width] - True where data is valid
    original_shape: Tuple[int, int]  # (original_height, original_width)
    padding: Tuple[int, int, int, int]  # (top, bottom, left, right)


class SS13MapTokenizer:
    def __init__(self, token_file_path: str, max_layers: int = 16):
        """
        Initialize tokenizer with token mapping and layer configuration

        Args:
            token_file_path: Path to JSON file with token mappings
            max_layers: Maximum number of layers to support
        """
        with open(token_file_path, "r") as f:
            token_data = json.load(f)

        self.reserved_tokens = token_data["reserved_tokens"]
        self.token_to_id = token_data["token_to_id"]
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.max_layers = max_layers

        # Reserved token IDs
        self.EMPTY_ID = self.reserved_tokens["<EMPTY>"]
        self.UNK_ID = self.reserved_tokens["<UNK>"]
        self.PAD_ID = self.reserved_tokens["<PAD>"]
        self.MASK_ID = self.reserved_tokens["<MASK>"]

        # Simple layer assignment with area exception
        self.turf_pattern = re.compile(r"^/turf/")
        self.area_pattern = re.compile(r"^/area/")

    def get_object_layer(self, obj_path: str, used_layers: set) -> Optional[int]:
        # Areas always go on the last layer (reserved, never dropped)
        if self.area_pattern.match(obj_path):
            return self.max_layers - 1

        for layer in range(0, self.max_layers - 1):
            if layer not in used_layers:
                return layer

        # No space available - will be dropped
        return None

    def _detect_tile_id_length(self, map_content: str) -> int:
        """Detect the length of tile IDs used in this map"""
        # Look for tile definitions to determine ID length
        tile_def_match = re.search(r'"([^"]+)"\s*=\s*\(', map_content)
        if tile_def_match:
            return len(tile_def_match.group(1))
        return 1  # Default fallback

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

    def parse_map_grid(self, map_content: str) -> List[List[str]]:
        """Parse the map grid from DMM content"""
        tile_length = self._detect_tile_id_length(map_content)
        lines = map_content.strip().split("\n")
        map_grid = []

        i = 0
        while i < len(lines):
            line = lines[i].strip()
            grid_match = re.match(r'\((\d+),(\d+),(\d+)\) = \{"', line)
            if grid_match:
                x_coord = int(grid_match.group(1))
                grid_content = ""

                i += 1
                while i < len(lines):
                    grid_line = lines[i].strip()
                    if grid_line == '"}':
                        break
                    if grid_line and not grid_line.startswith("("):
                        clean_line = grid_line.strip('"')
                        grid_content += clean_line
                    i += 1

                # Parse tiles based on detected length
                column_tiles = []
                for j in range(0, len(grid_content), tile_length):
                    if j + tile_length <= len(grid_content):
                        tile_id = grid_content[j : j + tile_length]
                        if tile_id.strip():
                            column_tiles.append(tile_id)

                # Extend map_grid if needed
                while len(map_grid) < x_coord:
                    map_grid.append([])

                map_grid[x_coord - 1] = column_tiles
            i += 1

        # Convert to row-major format
        if map_grid:
            max_height = max(len(col) for col in map_grid)
            transposed_grid = []
            for row_idx in range(max_height):
                row = []
                for col in map_grid:
                    if row_idx < len(col):
                        row.append(col[row_idx])
                    else:
                        # Use first available tile as default
                        default_tile = (
                            list(self.parse_tile_definitions(map_content).keys())[0]
                            if self.parse_tile_definitions(map_content)
                            else "aa"
                        )
                        row.append(default_tile)
                transposed_grid.append(row)
            map_grid = transposed_grid

        return map_grid

    def parse_dmm_map(self, map_string: str) -> Dict:
        """Parse DMM format map string using improved methods"""
        tile_definitions = self.parse_tile_definitions(map_string)
        map_grid = self.parse_map_grid(map_string)
        return {"tile_definitions": tile_definitions, "map_grid": map_grid}

    def tokenize_map(
        self,
        map_string: str,
        target_size: Optional[Tuple[int, int]] = None,
        pad_to_multiple: Optional[int] = None,
    ) -> MapTensor:
        """
        Tokenize a map string into layered tensor format

        Args:
            map_string: DMM format map string
            target_size: (height, width) to pad/crop to. If None, use original size
            pad_to_multiple: Pad dimensions to be multiple of this value

        Returns:
            MapTensor with tokenized data and metadata
        """
        parsed = self.parse_dmm_map(map_string)
        tile_definitions = parsed["tile_definitions"]
        map_grid = parsed["map_grid"]

        if not map_grid:
            raise ValueError("No map grid found in input")

        original_height = len(map_grid)
        original_width = len(map_grid[0])

        # Determine final dimensions
        if target_size:
            final_height, final_width = target_size
        elif pad_to_multiple:
            final_height = (
                (original_height + pad_to_multiple - 1) // pad_to_multiple
            ) * pad_to_multiple
            final_width = (
                (original_width + pad_to_multiple - 1) // pad_to_multiple
            ) * pad_to_multiple
        else:
            final_height, final_width = original_height, original_width

        # Initialize tensors
        data = np.full(
            (self.max_layers, final_height, final_width), self.PAD_ID, dtype=np.int32
        )
        mask = np.zeros((self.max_layers, final_height, final_width), dtype=bool)

        # Calculate padding
        pad_top = (final_height - original_height) // 2
        pad_left = (final_width - original_width) // 2
        pad_bottom = final_height - original_height - pad_top
        pad_right = final_width - original_width - pad_left

        # Fill in the actual map data
        for y in range(original_height):
            for x in range(original_width):
                final_y = y + pad_top
                final_x = x + pad_left

                tile_char = map_grid[y][x]
                if tile_char in tile_definitions:
                    objects = list(reversed(tile_definitions[tile_char]))

                    # Track which layers have been used for this tile
                    used_layers = set()
                    dropped_objects = []

                    for obj_path in objects:
                        if obj_path in self.token_to_id:
                            token_id = self.token_to_id[obj_path]
                            layer = self.get_object_layer(obj_path, used_layers)

                            if layer is not None:
                                data[layer, final_y, final_x] = token_id
                                mask[layer, final_y, final_x] = True
                                used_layers.add(layer)
                            else:
                                dropped_objects.append(obj_path)
                        else:
                            # Handle unknown objects
                            layer = self.get_object_layer("<UNK>", used_layers)
                            if layer is not None:
                                data[layer, final_y, final_x] = self.UNK_ID
                                mask[layer, final_y, final_x] = True
                                used_layers.add(layer)
                            else:
                                dropped_objects.append(obj_path)

                    if dropped_objects:
                        print(
                            f"Warning: Dropped {len(dropped_objects)} objects at ({y},{x}): {dropped_objects[:3]}{'...' if len(dropped_objects) > 3 else ''}"
                        )

                    # Ensure empty layers are marked as EMPTY, not PAD (within valid area)
                    for layer in range(self.max_layers):
                        if layer not in used_layers:
                            data[layer, final_y, final_x] = self.EMPTY_ID
                            mask[layer, final_y, final_x] = True

        return MapTensor(
            data=data,
            mask=mask,
            original_shape=(original_height, original_width),
            padding=(pad_top, pad_bottom, pad_left, pad_right),
        )

    def create_edit_mask(
        self,
        map_tensor: MapTensor,
        edit_region: Optional[Tuple[int, int, int, int]] = None,
        edit_layers: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Create a mask for regions/layers that can be edited during training

        Args:
            map_tensor: Tokenized map tensor
            edit_region: (top, left, bottom, right) region to allow editing
            edit_layers: List of layer indices that can be edited

        Returns:
            Boolean mask of same shape as data - True where editing is allowed
        """
        edit_mask = np.zeros_like(map_tensor.data, dtype=bool)

        # Determine editable region
        if edit_region:
            top, left, bottom, right = edit_region
        else:
            # Default to original map area (excluding padding)
            top = map_tensor.padding[0]
            left = map_tensor.padding[2]
            bottom = map_tensor.data.shape[1] - map_tensor.padding[1]
            right = map_tensor.data.shape[2] - map_tensor.padding[3]

        # Determine editable layers
        if edit_layers is None:
            edit_layers = list(range(self.max_layers))

        # Set edit mask
        for layer in edit_layers:
            if layer < self.max_layers:
                edit_mask[layer, top:bottom, left:right] = True

        # Only allow editing where original mask is True (valid data)
        edit_mask = edit_mask & map_tensor.mask

        return edit_mask

    def apply_mask_tokens(
        self, map_tensor: MapTensor, edit_mask: np.ndarray, mask_ratio: float = 0.15
    ) -> MapTensor:
        """
        Apply MASK tokens for training (like BERT masking)

        Args:
            map_tensor: Input map tensor
            edit_mask: Where masking is allowed
            mask_ratio: Fraction of tokens to mask

        Returns:
            New MapTensor with some tokens replaced by MASK
        """
        masked_data = map_tensor.data.copy()

        # Find maskable positions (not EMPTY, PAD, or already MASK)
        maskable = (
            edit_mask
            & (map_tensor.data != self.EMPTY_ID)
            & (map_tensor.data != self.PAD_ID)
            & (map_tensor.data != self.MASK_ID)
        )

        maskable_indices = np.where(maskable)
        n_maskable = len(maskable_indices[0])

        if n_maskable > 0:
            n_to_mask = int(n_maskable * mask_ratio)
            mask_indices = np.random.choice(n_maskable, n_to_mask, replace=False)

            for idx in mask_indices:
                layer, y, x = (
                    maskable_indices[0][idx],
                    maskable_indices[1][idx],
                    maskable_indices[2][idx],
                )
                masked_data[layer, y, x] = self.MASK_ID

        return MapTensor(
            data=masked_data,
            mask=map_tensor.mask.copy(),
            original_shape=map_tensor.original_shape,
            padding=map_tensor.padding,
        )

    def detokenize_map(self, map_tensor: MapTensor) -> str:
        """
        Convert tokenized map back to DMM format

        Args:
            map_tensor: Tokenized map data

        Returns:
            DMM format string
        """
        # Extract original map region
        top, bottom, left, right = map_tensor.padding
        height = map_tensor.original_shape[0]
        width = map_tensor.original_shape[1]

        # Create tile definitions by collecting unique combinations
        tile_combinations = {}
        tile_counter = 0
        tile_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        grid = []

        for y in range(height):
            row = []
            for x in range(width):
                actual_y = y + top
                actual_x = x + left

                # Collect all non-empty objects for this tile
                tile_objects = []
                for layer in range(self.max_layers):
                    token_id = map_tensor.data[layer, actual_y, actual_x]
                    if (
                        token_id != self.EMPTY_ID
                        and token_id != self.PAD_ID
                        and token_id != self.MASK_ID
                        and token_id in self.id_to_token
                    ):
                        tile_objects.append(self.id_to_token[token_id])

                # Create tile key
                tile_key = tuple(sorted(tile_objects))

                if tile_key not in tile_combinations:
                    if tile_counter < len(tile_chars):
                        tile_combinations[tile_key] = tile_chars[tile_counter]
                        tile_counter += 1
                    else:
                        # Fallback for too many unique tiles
                        tile_combinations[tile_key] = "z"

                row.append(tile_combinations[tile_key])
            grid.append(row)

        # Generate DMM string
        dmm_lines = []

        # Tile definitions
        for tile_objects, tile_char in tile_combinations.items():
            if tile_objects:  # Skip empty tiles for now
                dmm_lines.append(f'"{tile_char}" = (')
                for obj in tile_objects:
                    dmm_lines.append(f"{obj},")
                dmm_lines.append(")")

        # Map grid
        for x in range(width):
            column = [grid[y][x] for y in range(height)]
            dmm_lines.append(f'({x+1},1,1) = {{"')
            for char in column:
                dmm_lines.append(char)
            dmm_lines.append('"}')

        return "\n".join(dmm_lines)

    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        return len(self.token_to_id)


if __name__ == "__main__":
    tokenizer = SS13MapTokenizer("tiles_aleph.json", max_layers=8)

    map_data = """
"aa" = (
/obj/item/banner/cargo/mundane,
/obj/machinery/light/small{
        dir = 8
        },
/turf/open/floor/plating,
/area/template_noop)
"ba" = (
/turf/open/floor/plating,
/area/template_noop)
"ca" = (
/obj/item/banner/cargo/mundane,
/turf/open/floor/plating,
/area/template_noop)
"da" = (
/obj/effect/decal/cleanable/molten_object/large,
/turf/open/floor/plating,
/area/template_noop)
"ea" = (
/obj/effect/decal/cleanable/oil,
/turf/open/floor/plating,
/area/template_noop)
"fa" = (
/obj/structure/table,
/obj/item/candle{
        pixel_x = -5
        },
/obj/item/candle{
        pixel_x = 5
        },
/obj/item/fakeartefact,
/obj/effect/turf_decal/arrows,
/turf/open/floor/plating,
/area/template_noop)
(1,1,1) = {"
aa
ba
ea
"}
(2,1,1) = {"
ba
ba
ba
"}
(3,1,1) = {"
ca
da
fa
"}
"""

    # Tokenize the map
    print("=== SS13 Map Tokenization ===")

    # Basic tokenization
    map_tensor = tokenizer.tokenize_map(map_data)
    print(f"Original shape: {map_tensor.original_shape}")
    print(f"Tensor shape: {map_tensor.data.shape}")
    print(f"Padding: {map_tensor.padding}")

    # Show layer contents
    print("\n--- Layer Contents ---")
    for layer in range(tokenizer.max_layers):
        layer_data = map_tensor.data[layer]
        unique_tokens = np.unique(layer_data[map_tensor.mask[layer]])
        if len(unique_tokens) > 1 or (
            len(unique_tokens) == 1 and unique_tokens[0] != tokenizer.EMPTY_ID
        ):
            print(
                f"Layer {layer}: {[tokenizer.id_to_token.get(tid, f'ID_{tid}') for tid in unique_tokens]}"
            )
            print(f"  Data:\n{layer_data}")

    # Pad to 10x10 for training
    print(f"\n--- Padding to 10x10 ---")
    padded_tensor = tokenizer.tokenize_map(map_data, target_size=(10, 10))
    print(f"Padded shape: {padded_tensor.data.shape}")
    print(f"Padding applied: {padded_tensor.padding}")

    # Create edit mask
    edit_mask = tokenizer.create_edit_mask(padded_tensor, edit_layers=[0, 1, 2, 3])
    print(f"Edit mask shape: {edit_mask.shape}")
    print(f"Editable positions: {np.sum(edit_mask)}")

    # Apply masking for training
    masked_tensor = tokenizer.apply_mask_tokens(
        padded_tensor, edit_mask, mask_ratio=0.2
    )
    print(f"Masked tokens: {np.sum(masked_tensor.data == tokenizer.MASK_ID)}")

    print(f"\nVocabulary size: {tokenizer.get_vocab_size()}")

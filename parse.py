import re
from typing import Any, Dict, List, Tuple

import numpy as np


class SS13MapParser:
    def __init__(self):
        # Object type mappings - assign numerical IDs to different object types
        self.object_types = {
            "empty": 0,
            "/turf/open/floor/plating": 1,  # Floor
            "/obj/machinery/light/small": 2,  # Light
            "/obj/item/banner/cargo/mundane": 3,  # Banner
            "/obj/effect/decal/cleanable/molten_object/large": 4,  # Molten debris
            "/obj/effect/decal/cleanable/oil": 5,  # Oil stain
            "/obj/structure/table": 6,  # Table
            "/obj/item/candle": 7,  # Candle
            "/obj/item/fakeartefact": 8,  # Artifact
            "/obj/effect/turf_decal/arrows": 9,  # Arrow decal
            "/area/template_noop": 10,  # Area designation
        }

        self.tile_definitions = {}
        self.map_grid = []

    def parse_map_string(self, map_string: str) -> Dict[str, Any]:
        """Parse the SS13 map format string"""
        lines = map_string.strip().split("\n")

        # Parse tile definitions
        current_tile = None
        in_definition = False
        brace_depth = 0

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Check for tile definition start (e.g., '"a" = (')
            tile_match = re.match(r'"([a-z])" = \(', line)
            if tile_match:
                current_tile = tile_match.group(1)
                self.tile_definitions[current_tile] = []
                in_definition = True
                brace_depth = 0
                i += 1
                continue

            # Parse objects within definition
            if in_definition and current_tile:
                # Handle multi-line object definitions with parameters
                if line.startswith("/"):
                    obj_line = line.rstrip(",").strip()
                    # Handle objects with parameters in braces
                    if "{" in obj_line and "}" not in obj_line:
                        # Multi-line object definition
                        base_obj = obj_line.split("{")[0]
                        self.tile_definitions[current_tile].append(base_obj)
                        # Skip parameter lines until we find the closing brace
                        i += 1
                        while i < len(lines) and "}" not in lines[i]:
                            i += 1
                        i += 1  # Skip the closing brace line
                        continue
                    elif "{" in obj_line and "}" in obj_line:
                        # Single line object with parameters
                        base_obj = obj_line.split("{")[0]
                        self.tile_definitions[current_tile].append(base_obj)
                    else:
                        # Simple object without parameters
                        self.tile_definitions[current_tile].append(obj_line)
                elif line == ")":
                    # End of tile definition
                    in_definition = False
                    current_tile = None

            # Parse map grid - look for coordinate definitions
            grid_match = re.match(r'\((\d+),(\d+),(\d+)\) = \{"', line)
            if grid_match:
                x_coord = int(grid_match.group(1))
                # Extract grid content from this line and subsequent lines
                grid_content = []

                # Look for the opening quote
                if '{"' in line:
                    # Multi-line grid definition
                    i += 1
                    while i < len(lines):
                        grid_line = lines[i].strip()
                        if grid_line == '"}':
                            break
                        if grid_line and not grid_line.startswith("("):
                            grid_content.append(grid_line)
                        i += 1

                    # Store grid column
                    if len(self.map_grid) < x_coord:
                        self.map_grid.extend(
                            [[] for _ in range(x_coord - len(self.map_grid))]
                        )

                    if len(self.map_grid) >= x_coord:
                        self.map_grid[x_coord - 1] = grid_content

            i += 1

        # Convert map_grid from column-major to row-major format
        if self.map_grid:
            # Transpose the grid
            max_height = max(len(col) for col in self.map_grid)
            transposed_grid = []
            for row_idx in range(max_height):
                row = []
                for col in self.map_grid:
                    if row_idx < len(col):
                        row.append(col[row_idx])
                    else:
                        row.append("b")  # Default to empty floor
                transposed_grid.append(row)
            self.map_grid = transposed_grid

        return {
            "tile_definitions": self.tile_definitions,
            "map_grid": self.map_grid,
            "object_types": self.object_types,
        }

    def create_feature_tensors(self) -> np.ndarray:
        """Create multi-layer tensor where each layer represents presence of different object types"""
        if not self.map_grid:
            print("Warning: No map grid data found!")
            return np.array([])

        height = len(self.map_grid)
        width = len(self.map_grid[0]) if self.map_grid else 0
        num_features = len(self.object_types)

        print(
            f"Creating tensor with dimensions: {num_features} features x {height} height x {width} width"
        )

        # Initialize tensor: [features, height, width]
        tensor = np.zeros((num_features, height, width), dtype=int)

        for y, row in enumerate(self.map_grid):
            for x, tile_char in enumerate(row):
                if tile_char in self.tile_definitions:
                    tile_objects = self.tile_definitions[tile_char]

                    # Mark presence of each object type in corresponding feature layer
                    for obj in tile_objects:
                        if obj in self.object_types:
                            feature_id = self.object_types[obj]
                            tensor[feature_id, y, x] = 1

        return tensor

    def create_single_layer_tensor(
        self, priority_mapping: Dict[str, int] = None
    ) -> np.ndarray:
        """Create single layer tensor with priority-based object assignment"""
        if not self.map_grid:
            return np.array([])

        # Default priority mapping (higher numbers = higher priority)
        if priority_mapping is None:
            priority_mapping = {
                "/turf/open/floor/plating": 1,  # Base floor
                "/obj/effect/decal/cleanable/oil": 2,  # Oil stain
                "/obj/effect/decal/cleanable/molten_object/large": 3,  # Debris
                "/obj/machinery/light/small": 4,  # Light fixture
                "/obj/item/banner/cargo/mundane": 5,  # Banner
                "/obj/structure/table": 6,  # Table
                "/obj/item/candle": 7,  # Items on table
                "/obj/item/fakeartefact": 8,  # Artifact
                "/obj/effect/turf_decal/arrows": 9,  # Decals
            }

        height = len(self.map_grid)
        width = len(self.map_grid[0]) if self.map_grid else 0

        # Initialize with empty spaces
        tensor = np.zeros((height, width), dtype=int)

        for y, row in enumerate(self.map_grid):
            for x, tile_char in enumerate(row):
                if tile_char in self.tile_definitions:
                    tile_objects = self.tile_definitions[tile_char]

                    # Find highest priority object for this tile
                    max_priority = 0
                    selected_id = 0

                    for obj in tile_objects:
                        if obj in priority_mapping:
                            priority = priority_mapping[obj]
                            if priority > max_priority:
                                max_priority = priority
                                selected_id = self.object_types.get(obj, 0)

                    tensor[y, x] = selected_id

        return tensor

    def print_analysis(self):
        """Print detailed analysis of the parsed map"""
        print("=== SS13 Map Analysis ===")
        print(
            f"Map dimensions: {len(self.map_grid)}x{len(self.map_grid[0]) if self.map_grid else 0}"
        )
        print(f"Tile types defined: {len(self.tile_definitions)}")

        print("\n--- Tile Definitions ---")
        for tile_char, objects in self.tile_definitions.items():
            print(f"'{tile_char}': {len(objects)} objects")
            for obj in objects:
                obj_id = self.object_types.get(obj, "UNKNOWN")
                print(f"  - {obj} [ID: {obj_id}]")

        print("\n--- Map Grid ---")
        for i, row in enumerate(self.map_grid):
            print(f"Row {i}: {''.join(row)}")

        print(f"\n--- Object Type Registry ---")
        for obj_type, type_id in sorted(self.object_types.items(), key=lambda x: x[1]):
            print(f"ID {type_id}: {obj_type}")


# Example usage with your map data
map_data = """
"a" = (
/obj/item/banner/cargo/mundane,
/obj/machinery/light/small{
	dir = 8
	},
/turf/open/floor/plating,
/area/template_noop)
"b" = (
/turf/open/floor/plating,
/area/template_noop)
"c" = (
/obj/item/banner/cargo/mundane,
/turf/open/floor/plating,
/area/template_noop)
"d" = (
/obj/effect/decal/cleanable/molten_object/large,
/turf/open/floor/plating,
/area/template_noop)
"e" = (
/obj/effect/decal/cleanable/oil,
/turf/open/floor/plating,
/area/template_noop)
"f" = (
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
a
b
e
"}
(2,1,1) = {"
b
b
b
"}
(3,1,1) = {"
c
d
f
"}
"""

# Parse the map
parser = SS13MapParser()
result = parser.parse_map_string(map_data)

# Print analysis
parser.print_analysis()

# Create multi-layer feature tensor
feature_tensor = parser.create_feature_tensors()
print(f"\n--- Multi-layer Feature Tensor ---")
print(f"Shape: {feature_tensor.shape} (features, height, width)")
print("Sample layers:")
print("Floor layer (ID 1):")
print(feature_tensor[1])
print("\nBanner layer (ID 3):")
print(feature_tensor[3])

# Create single-layer tensor
single_tensor = parser.create_single_layer_tensor()
print(f"\n--- Single Layer Tensor ---")
print(f"Shape: {single_tensor.shape}")
print("Values (highest priority object per tile):")
print(single_tensor)

# Example: Create custom wall/empty binary tensor
wall_tensor = np.where(feature_tensor[1] == 1, 1, 0)  # 1 for floor, 0 for empty
print(f"\n--- Binary Floor/Empty Tensor ---")
print(wall_tensor)

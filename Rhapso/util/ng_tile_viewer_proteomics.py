#!/usr/bin/env python3
"""
Neuroglancer Tile Configuration Generator

Creates Neuroglancer configuration showing individual tiles with their
BigStitcher transformations applied, using actual data from S3.
"""

import json
import numpy as np
from tile_analyzer import BigStitcherAnalyzer
from collections import defaultdict

def voxel_to_meter_matrix(matrix_4x4, voxel_size):
    """
    Convert a 4x4 voxel-space transform (local_vox -> global_vox)
    to a 3x4 Neuroglancer-friendly matrix mapping local_vox -> world_meters.

    BigStitcher stores transforms in voxel units (x,y,z). Neuroglancer expects
    coordinates in meters. The conversion is a left-multiply by voxel scaling:
        world_m = V * (global_vox)
    where V = diag([vx_m, vy_m, vz_m, 1]).
    The returned value is the first 3 rows of V @ matrix_4x4 (a 3x4 list).
    """
    vx_m = voxel_size[0] * 1e-6
    vy_m = voxel_size[1] * 1e-6
    vz_m = voxel_size[2] * 1e-6
    V = np.diag([vx_m, vy_m, vz_m, 1.0])  # 4x4
    combined_m = V @ np.array(matrix_4x4, dtype=float)  # 4x4 in meters
    # Neuroglancer wants a 3x4 matrix (list-of-lists)
    return combined_m


class NeuroglancerTileConfig:
    # NOTE: Added optional legacy per-tile layer mode to restore original color implementation (no tint shaders, one layer per tile)
    def __init__(self, analyzer: BigStitcherAnalyzer, base_path: str,
                 show_correlations: bool = False,
                 name_with_avg_corr: bool = False,
                 quadrant_filter: str = None, nominal_only: bool = False,
                 legacy_colors: bool = False):
        self.analyzer = analyzer
        self.base_path = base_path.rstrip('/')
        self.show_correlations = show_correlations
        self.name_with_avg_corr = name_with_avg_corr
        self.quadrant_filter = quadrant_filter  # TL/TR/BL/BR/ALL/None
        self.nominal_only = nominal_only  # If True, only apply nominal translation (no affine corrections)
        self.legacy_colors = legacy_colors  # If True, revert to original behavior: one layer per tile, default Neuroglancer coloring

        first_tile = next(iter(self.analyzer.tiles.values()))
        self.voxel_size = list(first_tile.voxel_size)

        # Aggregated quadrant/color mode parameters (unused in legacy mode)
        self.color_one = "#00ff00"  # Green
        self.color_two = "#ff0000"  # Red
        self.layer_order = [
            ("TL", "Green"), ("TL", "Red"),
            ("TR", "Green"), ("TR", "Red"),
            ("BL", "Green"), ("BL", "Red"),
            ("BR", "Green"), ("BR", "Red"),
        ]
        self.layer_index_map = {qc: i for i, qc in enumerate(self.layer_order)}
        self.layers_created = {}

        self._calculate_volume_bounds()
        self.tile_rc = {}
        self._build_grid_index()

    def _build_grid_index(self):
        """Derive (row,col) indices from nominal tile positions instead of hard-coding 7."""
        xs = sorted({ int(round(t.nominal_position[0])) for t in self.analyzer.tiles.values() if t.nominal_position })
        ys = sorted({ int(round(t.nominal_position[1])) for t in self.analyzer.tiles.values() if t.nominal_position })
        x_to_col = {x:i for i,x in enumerate(xs)}
        y_to_row = {y:i for i,y in enumerate(ys)}
        self.grid_cols = len(xs)
        self.grid_rows = len(ys)
        self.tile_rc = {}
        for sid, tile in self.analyzer.tiles.items():
            if not tile.nominal_position: 
                continue
            x,y,_ = tile.nominal_position
            # Standard mapping: rows correspond to Y, columns correspond to X
            self.tile_rc[sid] = (y_to_row[int(round(y))], x_to_col[int(round(x))])
    
    def _calculate_volume_bounds(self):
        min_bounds = np.array([float('inf'), float('inf'), float('inf')])
        max_bounds = np.array([float('-inf'), float('-inf'), float('-inf')])
        for setup_id, tile in self.analyzer.tiles.items():
            if not tile.nominal_position:
                continue
            tile_transform = self._get_combined_transform(setup_id)
            corners = self._get_transformed_tile_corners(tile, tile_transform)
            tile_min = np.min(corners, axis=0)
            tile_max = np.max(corners, axis=0)
            min_bounds = np.minimum(min_bounds, tile_min)
            max_bounds = np.maximum(max_bounds, tile_max)
        self.volume_min = min_bounds
        self.volume_max = max_bounds
        self.volume_size = max_bounds - min_bounds
        print(f"Overall volume bounds: {min_bounds} to {max_bounds}")
        print(f"Volume size: {self.volume_size}")

    def _get_combined_transform(self, setup_id: int) -> np.ndarray:
        """
        Combine BigStitcher transforms for a tile (voxel units, x,y,z).
        Correct composition: apply affine correction first, then nominal placement.
        That is: p_world_vox = T_nominal @ A_affine @ p_local_vox
        We return the 4x4 combined matrix in voxel units.
        """
        tile = self.analyzer.tiles[setup_id]
        if not tile.transforms:
            raise ValueError(f"No transforms found for tile {tile.name}")

        transforms = tile.transforms

        if not len(transforms):
            raise ValueError(f"No transforms found for tile {tile.name}")
        
        # Find nominal and affine transforms by name (case-sensitive as in BigStitcher)

        # Applying transforms in reverse order
        # Transform order mat = local -> affine -> nominal
        if self.nominal_only:
            transforms = [t for t in transforms if "Translation to Nominal Grid" in t["name"]]
        
        transforms.reverse()
        combined = np.eye(4)

        for t in transforms:
            combined = np.array(t["matrix"], dtype=float) @ combined

        return combined

    def _get_transformed_tile_corners(self, tile, transform):
        # size_x, size_y, size_z = tile.size
        # # Use voxel coordinates [0 .. size-1] as corners (safer). Keep original behavior if you prefer full-size.
        # corners_local = np.array([
        #     [0, 0, 0, 1],
        #     [size_x - 1, 0, 0, 1],
        #     [0, size_y - 1, 0, 1],
        #     [0, 0, size_z - 1, 1],
        #     [size_x - 1, size_y - 1, 0, 1],
        #     [size_x - 1, 0, size_z - 1, 1],
        #     [0, size_y - 1, size_z - 1, 1],
        #     [size_x - 1, size_y - 1, size_z - 1, 1]
        # ], dtype=float)
        # # transform is returned in voxel units -> produces global voxel coordinates
        # corners_transformed = (np.array(transform, dtype=float) @ corners_local.T).T
        # return corners_transformed[:, :3]
        size_x, size_y, size_z = tile.size
        # Use full size as upper bounds (BigDataViewer convention)
        corners_local = np.array([
            [0, 0, 0, 1],
            [size_x, 0, 0, 1],
            [0, size_y, 0, 1],
            [0, 0, size_z, 1],
            [size_x, size_y, 0, 1],
            [size_x, 0, size_z, 1],
            [0, size_y, size_z, 1],
            [size_x, size_y, size_z, 1]
        ], dtype=float)
        corners_transformed = (np.array(transform, dtype=float) @ corners_local.T).T
        return corners_transformed[:, :3]

    def _voxels_to_physical(self, voxel_coords):
        return [
            voxel_coords[0] * self.voxel_size[0] * 1e-6,
            voxel_coords[1] * self.voxel_size[1] * 1e-6,
            voxel_coords[2] * self.voxel_size[2] * 1e-6
        ]

    def _create_tile_transform_matrix(self, setup_id: int, only_translation: bool = False):
        """
        Return a Neuroglancer-ready 3x4 matrix (list of lists) mapping
        local tile voxel coords -> world meters.
        """
        combined_transform_vox = self._get_combined_transform(setup_id)  # 4x4 in voxels
        # print("Before: ", combined_transform_vox)

        # For ZYX datasets, swap X and Z axes to convert to XYZ
        # combined_transform_vox[[0, 2], :] = combined_transform_vox[[2, 0], :]
        # combined_transform_vox[:, [0, 2]] = combined_transform_vox[:, [2, 0]]

        # print("Transposed matrix: ", combined_transform_vox)


        matrix = np.eye(4, dtype=float)
        if only_translation:
            matrix[:, -1] = combined_transform_vox[:, -1]
        
        else:
            matrix = combined_transform_vox

        
        # if self.analyzer.tiles[setup_id].name == "465720_509020.ome.zarr":
        #     print("Tile 465720_509020.ome.zarr combined transform (vox):")
        #     print(matrix)
        
        
        # print(matrix)
        # matrix = voxel_to_meter_matrix(matrix, self.voxel_size)
        # half_pixel_offset = np.eye(4)
        # half_pixel_offset[:3, 3] = 0.5
        # matrix = matrix @ half_pixel_offset
        
        return matrix[:3, :4].tolist()

    def _determine_quadrant(self, setup_id: int):
        """Return quadrant label (TL, TR, BL, BR) based on row/col."""
        row, col = self.tile_rc[setup_id]
        top = row < self.grid_rows / 2.0
        left = col < self.grid_cols / 2.0
        if top and left: return "TL"
        if top and not left: return "TR"
        if not top and left: return "BL"
        return "BR"

    def _color_bucket(self, setup_id: int):
        """Decide Green/Red bucket within quadrant (checkerboard logic retained)."""
        row, col = self.tile_rc[setup_id]
        return "Green" if (row + col) % 2 == 0 else "Red"

    def _make_shader(self, hex_color: str):
        r = int(hex_color[1:3], 16)/255.0
        g = int(hex_color[3:5], 16)/255.0
        b = int(hex_color[5:7], 16)/255.0
        return (
            "#uicontrol invlerp normalized\n"
            "#uicontrol float brightness slider(min=0,max=2,default=1)\n"
            "void main(){ vec3 c=vec3(%s,%s,%s); emitRGB(c*normalized()*brightness); }"
            % (r,g,b)
        )

    def _ensure_quadrant_color_layer(self, config: dict, quadrant: str, color_bucket: str, visible: bool, shader: str):
        key = (quadrant, color_bucket)
        if key in self.layers_created:
            return self.layer_index_map[key]

        idx = self.layer_index_map[key]
        # Decide final visibility: only the requested quadrant visible if filter set
        if self.quadrant_filter and self.quadrant_filter not in ("ALL", quadrant):
            layer_visible = False
        else:
            layer_visible = True  # user wants both color layers in selected quadrant visible

        # Layer name
        layer_name = f"{quadrant} - {color_bucket} Tiles"

        layer_obj = {
            "type": "image",
            "source": [],  # will append per-tile sources
            "name": layer_name,
            "visible": layer_visible,
            "opacity": 0.75,
            "shader": shader,
            "shaderControls": {
                "normalized": {"range": [0, 1200], "window": [100, 2000]},
                "brightness": 1.0
            },
            "blend": "additive"
        }

        # Insert in correct positional slot; pad list if needed
        while len(config["layers"]) <= idx:
            config["layers"].append({"_placeholder": True})
        config["layers"][idx] = layer_obj
        self.layers_created[key] = True
        return idx

    def _add_tile_layer(self, config, setup_id: int, tile):
        tile_path = f"{self.base_path}/{tile.name}"
        matrix_3x4 = self._create_tile_transform_matrix(setup_id)

        # Legacy mode: each tile gets its own layer, no shader tinting
        if self.legacy_colors:
            if "layers" not in config:
                config["layers"] = []
            quadrant = self._determine_quadrant(setup_id)
            if self.quadrant_filter and self.quadrant_filter not in ("ALL", quadrant):
                visible = False
            else:
                visible = True
            layer_obj = {
                "type": "image",
                "source": [
                    {
                        "url": f"zarr://{tile_path}",
                        "transform": {
                            "outputDimensions": {
                                "x": [self.voxel_size[0] * 1e-6, "m"],
                                "y": [self.voxel_size[1] * 1e-6, "m"],
                                "z": [self.voxel_size[2] * 1e-6, "m"],
                            },
                            "inputDimensions": {
                                "x": [self.voxel_size[0] * 1e-6, "m"],
                                "y": [self.voxel_size[1] * 1e-6, "m"],
                                "z": [self.voxel_size[2] * 1e-6, "m"],
                            },
                            "sourceRank": 3,
                            "matrix": matrix_3x4
                        }
                    }
                ],
                "name": tile.name if not self.name_with_avg_corr else f"{tile.name}",
                "visible": visible,
                "opacity": 1.0,
                "blend": "additive"
            }
            config["layers"].append(layer_obj)
            return

        # Aggregated quadrant/color mode
        quadrant = self._determine_quadrant(setup_id)
        color_bucket = self._color_bucket(setup_id)
        hex_color = self.color_one if color_bucket == "Green" else self.color_two
        shader = self._make_shader(hex_color)
        source_entry = {
            "url": f"zarr://{tile_path}",
            "transform": {
                "outputDimensions": {
                    "x": [self.voxel_size[0] * 1e-6, "m"],
                    "y": [self.voxel_size[1] * 1e-6, "m"],
                    "z": [self.voxel_size[2] * 1e-6, "m"],
                },
                "inputDimensions": {
                    "x": [self.voxel_size[0] * 1e-6, "m"],
                    "y": [self.voxel_size[1] * 1e-6, "m"],
                    "z": [self.voxel_size[2] * 1e-6, "m"],
                },
                "sourceRank": 3,
                "matrix": matrix_3x4
            }
        }
        if "layers" not in config:
            config["layers"] = []
        layer_idx = self._ensure_quadrant_color_layer(config, quadrant, color_bucket, True, shader)
        config["layers"][layer_idx]["source"].append(source_entry)

    def generate_config(self):
        center_voxel = self.volume_size / 2 + self.volume_min
        center_phys = self._voxels_to_physical(center_voxel)
        config = {
            "dimensions": {
                "x": [self.voxel_size[0] * 1e-6, "m"],
                "y": [self.voxel_size[1] * 1e-6, "m"],
                "z": [self.voxel_size[2] * 1e-6, "m"],
            },
            "position": center_phys,
            "crossSectionScale": 20.0,
            "projectionOrientation": [0.0, 0.0, 0.0, 1.0],
            "projectionScale": 2048.0,
            "layers": [],
            'layout': 'xy',
        }
        for setup_id, tile in self.analyzer.tiles.items():
            if not tile.nominal_position:
                continue
            self._add_tile_layer(config, setup_id, tile)
        if self.show_correlations and self.analyzer.stitching_pairs:
            self._add_correlation_annotations(config)
        return config

    def _compute_correlation_stats(self):
        stats = defaultdict(lambda: {"values": [], "neighbors": []})
        for pair in self.analyzer.stitching_pairs:
            a, b, corr = pair.setup_a, pair.setup_b, pair.correlation
            stats[a]["values"].append(corr)
            stats[a]["neighbors"].append((b, corr))
            stats[b]["values"].append(corr)
            stats[b]["neighbors"].append((a, corr))
        for tile_id, data in stats.items():
            vals = data["values"]
            if vals:
                data["mean"] = float(np.mean(vals))
                data["min"] = float(np.min(vals))
                data["max"] = float(np.max(vals))
                data["count"] = len(vals)
                data["neighbors"].sort(key=lambda x: x[1], reverse=True)
            else:
                data["mean"] = data["min"] = data["max"] = None
                data["count"] = 0
        return stats

    def _get_tile_avg_correlation(self, setup_id: int):
        if not hasattr(self, "_cached_corr_stats"):
            self._cached_corr_stats = self._compute_correlation_stats()
        data = self._cached_corr_stats.get(setup_id)
        if not data:
            return None
        return data.get("mean")

    def _add_correlation_annotations(self, config):
        corr_stats = self._compute_correlation_stats()
        annotations = []
        for setup_id, tile in self.analyzer.tiles.items():
            if not tile.nominal_position:
                continue
            transform = self._get_combined_transform(setup_id)
            center_local = np.array([tile.size[0]/2, tile.size[1]/2, tile.size[2]/2, 1])
            center_transformed = transform @ center_local
            center_phys = self._voxels_to_physical(center_transformed[:3])
            stats = corr_stats.get(setup_id, {})
            mean_corr = stats.get("mean")
            if mean_corr is None:
                desc = f"Tile {setup_id}: no stitching pairs"
            else:
                min_corr = stats.get("min")
                max_corr = stats.get("max")
                count = stats.get("count")
                neighbor_strs = [f"{nid}:{c:.3f}" for nid, c in stats.get("neighbors", [])]
                if len(neighbor_strs) > 8:
                    shown = neighbor_strs[:8]
                    shown.append(f"...(+{len(neighbor_strs)-8} more)")
                    neighbor_str = ', '.join(shown)
                else:
                    neighbor_str = ', '.join(neighbor_strs)
                desc = (f"Tile {setup_id} correlations\n"
                        f" n={count} mean={mean_corr:.4f} min={min_corr:.4f} max={max_corr:.4f}\n"
                        f" neighbors: {neighbor_str}")
            annotations.append({"type": "point", "id": f"corr_{setup_id}", "point": center_phys, "description": desc})
        config["layers"].append({
            "type": "annotation",
            "name": "Tile Correlations",
            "annotations": annotations,
            "annotationColor": "#ffaa00",
            "visible": True,
            "description": "Per-tile stitching correlation statistics"
        })

    def _get_tile_avg_correlation(self, setup_id: int):
        if not hasattr(self, "_cached_corr_stats"):
            self._cached_corr_stats = self._compute_correlation_stats()
        data = self._cached_corr_stats.get(setup_id)
        if not data:
            return None
        return data.get("mean")

    def save_config(self, config, filename):
        """Save configuration to file and create URL"""
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Neuroglancer tile config saved to: {filename}")

        # Create URL
        # import urllib.parse
        # config_str = json.dumps(config, separators=(',', ':'))
        # encoded_config = urllib.parse.quote(config_str)

        # neuroglancer_url = f"https://neuroglancer-demo.appspot.com/#!{encoded_config}"

        # # Save URL (may be truncated due to length)
        # url_filename = filename.replace('.json', '_url.txt')
        # with open(url_filename, 'w') as f:
        #     f.write(neuroglancer_url)

        # print(f"Neuroglancer URL saved to: {url_filename}")
        # print(f"Note: URL may be truncated due to length - use JSON import instead")

def main():
    # 🔧 Hardcoded parameters – edit these as needed
    xml_file = "/Users/sean.fite/Desktop/823476/HCR_823476_round3_solver.xml"
    output_path = "/Users/sean.fite/Desktop/HCR_823476_r3_ng_link.json"
    show_correlations = False          # True to include correlation annotations
    name_with_avg_corr = False         # True to append avg corr to layer names
    quadrant = "ALL"                   # One of: "ALL", "TL", "TR", "BL", "BR"
    nominal_only = False               # True to only apply nominal translation
    legacy_colors = False              # True for original per-tile layers

    print(f"Loading BigStitcher data from {xml_file}...")
    analyzer = BigStitcherAnalyzer(xml_file)
    analyzer.parse_tiles()
    analyzer.parse_stitching_results()
    print(f"Found {len(analyzer.tiles)} tiles")

    tile_config = NeuroglancerTileConfig(
        analyzer,
        analyzer.base_path,
        show_correlations=show_correlations,
        name_with_avg_corr=name_with_avg_corr,
        quadrant_filter=quadrant,
        nominal_only=nominal_only,
        legacy_colors=legacy_colors,
    )

    if legacy_colors:
        print("Generating original per-tile layers (legacy color mode)...")
    else:
        print("Generating 8 quadrant-color aggregated layers (TL/TR/BL/BR × Green/Red)...")

    config = tile_config.generate_config()
    tile_config.save_config(config, output_path)

    print("\n=== Configuration Summary ===")
    if legacy_colors:
        print(f"Legacy mode: {len(config['layers'])} per-tile layers created.")
        if quadrant and quadrant != "ALL":
            visible_count = sum(1 for L in config["layers"] if L.get("visible"))
            print(f"Only quadrant {quadrant} tile layers visible ({visible_count} layers).")
    else:
        print("Created 8 aggregated layers (quadrant × color).")
        if quadrant and quadrant != "ALL":
            print(f"Only quadrant {quadrant} layers set visible.")
        else:
            print("All quadrant layers visible.")

    if show_correlations:
        print("Correlation annotation layer included.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()

# source .venv/bin/activate 
# python -m Rhapso.eval.ng_tile_viewer_proteomics
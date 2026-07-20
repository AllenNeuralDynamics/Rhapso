#!/usr/bin/env python3
"""
BigStitcher XML Tile Analyzer

This script parses BigStitcher XML files and extracts comprehensive information
about each tile including resolution, transforms, and stitching results.
"""

import xml.etree.ElementTree as ET
import numpy as np
import argparse
import json
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import re

@dataclass
class TileInfo:
    """Container for tile information"""
    setup_id: int
    name: str
    size: Tuple[int, int, int]  # X, Y, Z in voxels
    voxel_size: Tuple[float, float, float]  # X, Y, Z in microns
    unit: str
    attributes: Dict[str, Any]
    transforms: List[Dict[str, Any]]
    nominal_position: Tuple[float, float, float] = None  # X, Y, Z in microns
    
@dataclass
class StitchingPair:
    """Container for pairwise stitching results"""
    setup_a: int
    setup_b: int
    timepoint_a: int
    timepoint_b: int
    shift_matrix: np.ndarray  # 4x4 transformation matrix
    correlation: float
    hash_value: float
    overlap_bbox: Tuple[float, float, float, float, float, float]  # x_min, y_min, z_min, x_max, y_max, z_max

class BigStitcherAnalyzer:
    """Analyzer for BigStitcher XML files"""
    
    def __init__(self, xml_path: str):
        """Initialize with path to BigStitcher XML file"""
        self.xml_path = xml_path
        self.tree = ET.parse(xml_path)
        self.root = self.tree.getroot()
        self.tiles = {}
        self.stitching_pairs = []
        self.base_path = self.root.find('.//zarr').text if self.root.find('.//zarr') is not None else ''
        
        if not len(self.base_path):
            raise ValueError("Base path for tile data not found in XML")
                                        
        
    def parse_tiles(self) -> Dict[int, TileInfo]:
        """Parse all tile information from ViewSetups and ViewRegistrations"""
        
        # First, parse ViewSetups for basic tile info
        view_setups = self.root.find('.//ViewSetups')
        if view_setups is None:
            raise ValueError("No ViewSetups found in XML")
            
        for view_setup in view_setups.findall('ViewSetup'):
            setup_id = int(view_setup.find('id').text)
            name = view_setup.find('name').text
            
            # Parse size (X, Y, Z)
            size_text = view_setup.find('size').text.strip()
            width, height, depth = map(int, size_text.split())
            
            # Parse voxel size
            voxel_size_elem = view_setup.find('voxelSize')
            unit = voxel_size_elem.find('unit').text
            voxel_text = voxel_size_elem.find('size').text.strip()
            vx, vy, vz = map(float, voxel_text.split())
            
            # Parse attributes
            attrs = {}
            attributes_elem = view_setup.find('attributes')
            if attributes_elem is not None:
                for attr in attributes_elem:
                    attrs[attr.tag] = attr.text
                    
            self.tiles[setup_id] = TileInfo(
                setup_id=setup_id,
                name=name,
                size=(width, height, depth),
                voxel_size=(vx, vy, vz),
                unit=unit,
                attributes=attrs,
                transforms=[]
            )
        
        # Parse ViewRegistrations for transforms
        view_registrations = self.root.find('.//ViewRegistrations')
        if view_registrations is not None:
            for view_reg in view_registrations.findall('ViewRegistration'):
                setup_id = int(view_reg.get('setup'))

                if setup_id not in self.tiles:
                    continue

                # Parse all transforms for this tile
                transforms = []
                saw_nominal = False
                acquisition_offset = None
                for view_transform in view_reg.findall('ViewTransform'):
                    transform_info = self._parse_transform(view_transform)
                    transforms.append(transform_info)

                    name = transform_info['name'] or ''
                    if "Translation to Nominal Grid" in name:
                        matrix = transform_info['matrix']
                        tx_voxels, ty_voxels, tz_voxels = matrix[0, 3], matrix[1, 3], matrix[2, 3]
                        self.tiles[setup_id].nominal_position = (tx_voxels, ty_voxels, tz_voxels)
                        saw_nominal = True
                    elif "Acquisition Offset" in name:
                        # Capture as potential fallback nominal position
                        m = transform_info['matrix']
                        acquisition_offset = (m[0,3], m[1,3], m[2,3])

                self.tiles[setup_id].transforms = transforms
                if not saw_nominal and acquisition_offset and not self.tiles[setup_id].nominal_position:
                    # Fallback: use acquisition offset as nominal position when explicit nominal translation missing.
                    self.tiles[setup_id].nominal_position = acquisition_offset
                    # print(f"[INFO] Fallback nominal position from Acquisition Offset for setup {setup_id}")

        # SECONDARY FALLBACK: derive nominal positions from tile names if still missing.
        # Expected patterns: '419730_488950.zarr', '465720_509020.ome.zarr'
        name_pattern = re.compile(r'^(\d+)_+(\d+)')
        for tid, tile in self.tiles.items():
            if tile.nominal_position is None:
                m = name_pattern.match(tile.name)
                if m:
                    x_val = float(m.group(1))
                    y_val = float(m.group(2))
                    tile.nominal_position = (x_val, y_val, 0.0)
                    # print(f"[INFO] Derived nominal position from name for setup {tid}: {tile.nominal_position}")
                elif tile.transforms:
                    # Last-resort: use translation components of first transform
                    mat = tile.transforms[-1]['matrix'] if tile.transforms[-1].get('matrix') is not None else np.eye(4)
                    tx, ty, tz = mat[0,3], mat[1,3], mat[2,3]
                    tile.nominal_position = (tx, ty, tz)
                    # print(f"[INFO] Derived nominal position from transform translation for setup {tid}: {tile.nominal_position}")
                
        return self.tiles
    
    def _parse_transform(self, view_transform) -> Dict[str, Any]:
        """Parse a single ViewTransform element"""
        transform_info = {
            'type': view_transform.get('type', 'unknown'),
            'name': None,
            'matrix': np.eye(4),
            'raw_values': None
        }
        
        # Parse name
        name_elem = view_transform.find('Name')
        # print("Parsing transform type:", transform_info['type'], "Name element:", name_elem.text)
        if name_elem is not None:
            transform_info['name'] = name_elem.text.strip()
            
        # Parse affine matrix
        affine_elem = view_transform.find('affine')
        if affine_elem is not None:
            affine_values = list(map(float, affine_elem.text.split()))
            if len(affine_values) == 12:
                # Convert 12-element affine to 4x4 matrix
                matrix = np.array([
                    [affine_values[0], affine_values[1], affine_values[2], affine_values[3]],
                    [affine_values[4], affine_values[5], affine_values[6], affine_values[7]],
                    [affine_values[8], affine_values[9], affine_values[10], affine_values[11]],
                    [0, 0, 0, 1]
                ])
                transform_info['matrix'] = matrix
                transform_info['raw_values'] = affine_values
                
        return transform_info

    def parse_stitching_results(self) -> List[StitchingPair]:
        """Parse stitching results for tile pairs"""
        stitching_results = self.root.find('.//StitchingResults')
        if stitching_results is None:
            return []
            
        pairs = []
        for pairwise_result in stitching_results.findall('PairwiseResult'):
            setup_a = int(pairwise_result.get('view_setup_a'))
            setup_b = int(pairwise_result.get('view_setup_b'))
            tp_a = int(pairwise_result.get('tp_a'))
            tp_b = int(pairwise_result.get('tp_b'))
            
            # Parse shift matrix
            shift_elem = pairwise_result.find('shift')
            shift_values = list(map(float, shift_elem.text.split()))
            shift_matrix = np.array([
                [shift_values[0], shift_values[1], shift_values[2], shift_values[3]],
                [shift_values[4], shift_values[5], shift_values[6], shift_values[7]],
                [shift_values[8], shift_values[9], shift_values[10], shift_values[11]],
                [0, 0, 0, 1]
            ])
            
            # Parse correlation
            corr_elem = pairwise_result.find('correlation')
            correlation = float(corr_elem.text) if corr_elem is not None else 0.0
            
            # Parse hash
            hash_elem = pairwise_result.find('hash')
            hash_value = float(hash_elem.text) if hash_elem is not None else 0.0
            
            # Parse overlap bounding box
            bbox_elem = pairwise_result.find('overlap_boundingbox')
            if bbox_elem is not None:
                bbox_values = list(map(float, bbox_elem.text.split()))
                overlap_bbox = tuple(bbox_values)
            else:
                overlap_bbox = (0, 0, 0, 0, 0, 0)
            
            pair = StitchingPair(
                setup_a=setup_a,
                setup_b=setup_b,
                timepoint_a=tp_a,
                timepoint_b=tp_b,
                shift_matrix=shift_matrix,
                correlation=correlation,
                hash_value=hash_value,
                overlap_bbox=overlap_bbox
            )
            pairs.append(pair)
            
        self.stitching_pairs = pairs
        return pairs
    
    def analyze_overlap_patterns(self) -> Dict[str, Any]:
        """Analyze tile overlap patterns and processing vs geometric overlap"""
        
        if not self.tiles or not self.stitching_pairs:
            return {}
            
        # Get tile specifications
        sample_tile = next(iter(self.tiles.values()))
        tile_size_x, tile_size_y, tile_size_z = sample_tile.size
        voxel_size_x, voxel_size_y, voxel_size_z = sample_tile.voxel_size
        tile_phys_x = tile_size_x * voxel_size_x
        tile_phys_y = tile_size_y * voxel_size_y
        tile_phys_z = tile_size_z * voxel_size_z
        
        # Find spacing patterns from nominal positions
        positions = [(sid, tile.nominal_position) for sid, tile in self.tiles.items() if tile.nominal_position]
        
        x_spacings = []
        y_spacings = []
        
        for i, (sid_a, pos_a) in enumerate(positions):
            for j, (sid_b, pos_b) in enumerate(positions[i+1:], i+1):
                dx = abs(pos_a[0] - pos_b[0])
                dy = abs(pos_a[1] - pos_b[1])
                
                # Only consider adjacent tiles
                if dx > 0 and dy < 1e-6:  # X-direction neighbors (allowing for small floating point errors)
                    x_spacings.append(dx)
                elif dy > 0 and dx < 1e-6:  # Y-direction neighbors  
                    y_spacings.append(dy)
        
        overlap_analysis = {
            'tile_size_um': (tile_phys_x, tile_phys_y, tile_phys_z),
            'tile_size_voxels': (tile_size_x, tile_size_y, tile_size_z),
            'voxel_size_um': (voxel_size_x, voxel_size_y, voxel_size_z)
        }
        
        if x_spacings:
            x_spacing_um = min(x_spacings)
            x_overlap_um = tile_phys_x - x_spacing_um
            x_overlap_pct = (x_overlap_um / tile_phys_x) * 100
            overlap_analysis['x_spacing_um'] = x_spacing_um
            overlap_analysis['x_overlap_um'] = x_overlap_um
            overlap_analysis['x_overlap_pct'] = x_overlap_pct
        
        if y_spacings:
            y_spacing_um = min(y_spacings)
            y_overlap_um = tile_phys_y - y_spacing_um
            y_overlap_pct = (y_overlap_um / tile_phys_y) * 100
            overlap_analysis['y_spacing_um'] = y_spacing_um
            overlap_analysis['y_overlap_um'] = y_overlap_um
            overlap_analysis['y_overlap_pct'] = y_overlap_pct
        
        # Analyze processing overlaps from stitching results
        # Separate pairs by their primary overlap direction
        x_direction_pairs = []  # Horizontally adjacent (X-overlap matters)
        y_direction_pairs = []  # Vertically adjacent (Y-overlap matters)
        
        for pair in self.stitching_pairs:
            # Get nominal positions to determine pair direction
            if pair.setup_a in self.tiles and pair.setup_b in self.tiles:
                pos_a = self.tiles[pair.setup_a].nominal_position
                pos_b = self.tiles[pair.setup_b].nominal_position
                
                if pos_a and pos_b:
                    dx = abs(pos_a[0] - pos_b[0])
                    dy = abs(pos_a[1] - pos_b[1])
                    
                    # Determine primary direction (allow small floating point errors)
                    if dx > 1e-6 and dy < 1e-6:  # X-direction neighbors (horizontal)
                        x_direction_pairs.append(pair)
                    elif dy > 1e-6 and dx < 1e-6:  # Y-direction neighbors (vertical)
                        y_direction_pairs.append(pair)
                    # Note: We ignore diagonal pairs for overlap analysis
        
        # Calculate processing overlaps for each direction
        if x_direction_pairs:
            x_proc_overlaps = []
            for pair in x_direction_pairs:
                x_min, y_min, z_min, x_max, y_max, z_max = pair.overlap_bbox
                proc_overlap_x = x_max - x_min
                proc_overlap_x_pct = (proc_overlap_x / tile_phys_x) * 100
                x_proc_overlaps.append(proc_overlap_x_pct)
            
            overlap_analysis['x_processing_overlap_avg'] = np.mean(x_proc_overlaps)
            overlap_analysis['x_processing_overlap_std'] = np.std(x_proc_overlaps)
            overlap_analysis['x_direction_pairs_count'] = len(x_direction_pairs)
            
        if y_direction_pairs:
            y_proc_overlaps = []
            for pair in y_direction_pairs:
                x_min, y_min, z_min, x_max, y_max, z_max = pair.overlap_bbox
                proc_overlap_y = y_max - y_min
                proc_overlap_y_pct = (proc_overlap_y / tile_phys_y) * 100
                y_proc_overlaps.append(proc_overlap_y_pct)
            
            overlap_analysis['y_processing_overlap_avg'] = np.mean(y_proc_overlaps)
            overlap_analysis['y_processing_overlap_std'] = np.std(y_proc_overlaps)
            overlap_analysis['y_direction_pairs_count'] = len(y_direction_pairs)
            
        # Calculate ratios between processing and geometric overlaps
        if 'x_overlap_pct' in overlap_analysis and 'x_processing_overlap_avg' in overlap_analysis:
            overlap_analysis['x_processing_to_geometric_ratio'] = overlap_analysis['x_processing_overlap_avg'] / overlap_analysis['x_overlap_pct']
            
        if 'y_overlap_pct' in overlap_analysis and 'y_processing_overlap_avg' in overlap_analysis:
            overlap_analysis['y_processing_to_geometric_ratio'] = overlap_analysis['y_processing_overlap_avg'] / overlap_analysis['y_overlap_pct']
            
        return overlap_analysis
    
    def print_overlap_summary(self):
        """Print a summary of overlap analysis"""
        overlap_info = self.analyze_overlap_patterns()
        
        if not overlap_info:
            print("No overlap information available")
            return
            
        print(f"\n=== Overlap Analysis Summary ===")
        print(f"Tile size: {overlap_info['tile_size_voxels'][0]} × {overlap_info['tile_size_voxels'][1]} × {overlap_info['tile_size_voxels'][2]} voxels")
        print(f"Physical tile size: {overlap_info['tile_size_um'][0]:.1f} × {overlap_info['tile_size_um'][1]:.1f} × {overlap_info['tile_size_um'][2]:.1f} µm")
        
        print(f"\nGeometric Overlap (Design):")
        if 'x_overlap_pct' in overlap_info:
            print(f"  X-direction: {overlap_info['x_overlap_um']:.1f} µm ({overlap_info['x_overlap_pct']:.1f}%)")
            
        if 'y_overlap_pct' in overlap_info:
            print(f"  Y-direction: {overlap_info['y_overlap_um']:.1f} µm ({overlap_info['y_overlap_pct']:.1f}%)")
        
        print(f"\nProcessing Overlap (Correlation window dimensions by direction):")
        if 'x_processing_overlap_avg' in overlap_info:
            # Convert to actual microns for the correlation window
            x_window_um = (overlap_info['x_processing_overlap_avg']/100) * overlap_info['tile_size_um'][0]
            x_pairs = overlap_info.get('x_direction_pairs_count', 0)
            print(f"  X-direction pairs ({x_pairs}): {x_window_um:.1f} µm window ({overlap_info['x_processing_overlap_avg']:.1f}% of tile)")
            
        if 'y_processing_overlap_avg' in overlap_info:
            # Convert to actual microns for the correlation window
            y_window_um = (overlap_info['y_processing_overlap_avg']/100) * overlap_info['tile_size_um'][1]
            y_pairs = overlap_info.get('y_direction_pairs_count', 0)
            print(f"  Y-direction pairs ({y_pairs}): {y_window_um:.1f} µm window ({overlap_info['y_processing_overlap_avg']:.1f}% of tile)")
        
        print(f"\nProcessing Analysis:")
        print(f"The 'overlap_boundingbox' shows the correlation search window used by BigStitcher.")
        
        if 'y_processing_overlap_avg' in overlap_info and 'y_overlap_pct' in overlap_info:
            # For Y direction (primary overlap direction in your data)
            y_window_um = (overlap_info['y_processing_overlap_avg']/100) * overlap_info['tile_size_um'][1]
            geometric_y_um = overlap_info['y_overlap_um']
            ratio = overlap_info.get('y_processing_to_geometric_ratio', 0)
            
            print(f"  Geometric Y overlap: {geometric_y_um:.1f} µm (10% design)")
            print(f"  Processing Y window: {y_window_um:.1f} µm (correlation search area)")
            print(f"  Window utilization: {ratio:.3f} ({ratio*100:.1f}% of geometric overlap)")
        
        if 'x_processing_overlap_avg' in overlap_info and 'x_overlap_pct' in overlap_info:
            # For X direction 
            x_window_um = (overlap_info['x_processing_overlap_avg']/100) * overlap_info['tile_size_um'][0]
            geometric_x_um = overlap_info['x_overlap_um']
            ratio = overlap_info.get('x_processing_to_geometric_ratio', 0)
            
            print(f"  Geometric X overlap: {geometric_x_um:.1f} µm (10% design)")
            print(f"  Processing X window: {x_window_um:.1f} µm (correlation search area)")
            print(f"  Window utilization: {ratio:.3f} ({ratio*100:.1f}% of geometric overlap)")
        
        print(f"\nKey insights:")
        print(f"  • BigStitcher separates X-direction and Y-direction tile pairs")
        print(f"  • Each pair type uses appropriate correlation windows for its overlap direction")
        print(f"  • Processing windows are conservative subsets of the geometric overlap")
        print(f"  • Edge regions are avoided to prevent imaging artifacts from affecting alignment")
        
    def get_tile_neighbors(self, setup_id: int) -> List[Tuple[int, StitchingPair]]:
        """Get all neighboring tiles for a given setup ID"""
        neighbors = []
        for pair in self.stitching_pairs:
            if pair.setup_a == setup_id:
                neighbors.append((pair.setup_b, pair))
            elif pair.setup_b == setup_id:
                neighbors.append((pair.setup_a, pair))
        return neighbors
    
    def analyze_transform_chain(self, setup_id: int) -> Dict[str, Any]:
        """Analyze and combine all transforms for a given tile following BigStitcher's composition order."""
        if setup_id not in self.tiles:
            return None

        tile = self.tiles[setup_id]
        transforms = tile.transforms
        transforms.reverse()

        analysis = {
            "total_transforms": len(transforms),
            "transform_names": [t["name"] for t in transforms],
            "individual_transforms": {},
            "combined_matrix": np.eye(4),
            "translation_components": [],
            "rotation_components": [],
            "has_nominal_translation": False,
            "has_interpolated_affine": False,
        }

        # Keep track of known transforms
        interpolated_transform = np.eye(4)
        nominal_transform = np.eye(4)

        # Analyze individual transforms
        for t in transforms:
            name = t.get("name", "Unnamed")
            mat = np.array(t["matrix"], dtype=float)

            # Extract translation and rotation
            translation = mat[:3, 3]
            rotation = mat[:3, :3]

            analysis["translation_components"].append(tuple(translation))
            analysis["rotation_components"].append(rotation)

            analysis["individual_transforms"][name] = {
                "matrix": mat,
                "translation": tuple(translation),
                "rotation_scale": rotation,
            }

            # Identify known types
            if "InterpolatedAffineModel3D" in name or "Stitching Transform" in name:
                interpolated_transform = mat
                analysis["has_interpolated_affine"] = True
            elif "Translation to Nominal Grid" in name:
                nominal_transform = mat
                analysis["has_nominal_translation"] = True

        # --- Compose full transform chain ---
        # BigStitcher applies local transforms first (affine), then nominal grid translation
        combined = np.eye(4)
        for t in transforms:
            combined = np.dot(t["matrix"], combined)

        analysis["combined_matrix"] = combined
        analysis["final_translation"] = tuple(combined[:3, 3])
        analysis["final_rotation_scale"] = combined[:3, :3]

        # --- Additional info ---
        analysis["composition_method"] = "Affine matrix multiplication (BigStitcher order)"
        analysis["composition_order"] = "Apply affine correction first, then nominal translation"

        # --- Optional explicit nominal x affine check ---
        if analysis["has_nominal_translation"] and analysis["has_interpolated_affine"]:
            combined_direct = nominal_transform @ interpolated_transform
            analysis["combined_direct_nominal_affine"] = combined_direct
            analysis["direct_final_translation"] = tuple(combined_direct[:3, 3])

        # --- Rotation/scale diagnostics ---
        R = combined[:3, :3]
        det = np.linalg.det(R)
        is_identity = np.allclose(R, np.eye(3), atol=1e-6)
        analysis["has_rotation_or_scaling"] = not is_identity
        analysis["rotation_analysis"] = {
            "determinant": det,
            "is_pure_rotation": np.isclose(det, 1.0, atol=1e-6),
            "rotation_matrix": R,
        }

        return analysis

    
    def export_to_csv(self, output_path: str):
        """Export tile information to CSV"""
        data = []
        for setup_id, tile in self.tiles.items():
            # Get neighbors
            neighbors = self.get_tile_neighbors(setup_id)
            neighbor_ids = [n[0] for n in neighbors]
            
            # Get transform analysis
            transform_analysis = self.analyze_transform_chain(setup_id)
            
            row = {
                'setup_id': setup_id,
                'name': tile.name,
                'size_x': tile.size[0],
                'size_y': tile.size[1],
                'size_z': tile.size[2],
                'voxel_size_x': tile.voxel_size[0],
                'voxel_size_y': tile.voxel_size[1],
                'voxel_size_z': tile.voxel_size[2],
                'unit': tile.unit,
                'nominal_pos_x': tile.nominal_position[0] if tile.nominal_position else None,
                'nominal_pos_y': tile.nominal_position[1] if tile.nominal_position else None,
                'nominal_pos_z': tile.nominal_position[2] if tile.nominal_position else None,
                'num_transforms': transform_analysis['total_transforms'],
                'transform_names': '; '.join(transform_analysis['transform_names']),
                'final_translation_x': transform_analysis['final_translation'][0],
                'final_translation_y': transform_analysis['final_translation'][1],
                'final_translation_z': transform_analysis['final_translation'][2],
                'num_neighbors': len(neighbors),
                'neighbor_ids': '; '.join(map(str, neighbor_ids)),
                **tile.attributes
            }
            data.append(row)
            
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Tile information exported to {output_path}")
    
    def export_stitching_to_csv(self, output_path: str, threshold: float = 0.0):
        """Export stitching results to CSV"""
        over_threshold = []
        under_threshold = []
        for pair in self.stitching_pairs:
            row = {
                'name_a': self.tiles[pair.setup_a].name if pair.setup_a in self.tiles else f"Setup {pair.setup_a}",
                'name_b': self.tiles[pair.setup_b].name if pair.setup_b in self.tiles else f"Setup {pair.setup_b}",
                'setup_a': pair.setup_a,
                'setup_b': pair.setup_b,
                'timepoint_a': pair.timepoint_a,
                'timepoint_b': pair.timepoint_b,
                'correlation': pair.correlation,
                'hash': pair.hash_value,
                'shift_tx': pair.shift_matrix[0, 3],
                'shift_ty': pair.shift_matrix[1, 3],
                'shift_tz': pair.shift_matrix[2, 3],
                'overlap_x_min': pair.overlap_bbox[0],
                'overlap_y_min': pair.overlap_bbox[1],
                'overlap_z_min': pair.overlap_bbox[2],
                'overlap_x_max': pair.overlap_bbox[3],
                'overlap_y_max': pair.overlap_bbox[4],
                'overlap_z_max': pair.overlap_bbox[5],
            }

            if float(pair.correlation) < float(threshold):
                under_threshold.append(row)
            else:
                over_threshold.append(row)

        df_over_threshold = pd.DataFrame(over_threshold)
        df_over_threshold.to_csv(output_path, index=False)
        print(f"Stitching results over threshold {threshold} exported to {output_path}")

        if len(under_threshold) > 0:
            output_path_under = output_path.replace(".csv", f"_under_threshold_{threshold}.csv")
            df_under_threshold = pd.DataFrame(under_threshold)
            df_under_threshold.to_csv(output_path_under, index=False)
            print(f"Stitching results under threshold {threshold} exported to {output_path_under}")

        return over_threshold, under_threshold
    
    def print_summary(self):
        """Print a summary of the analysis"""
        print(f"\n=== BigStitcher XML Analysis Summary ===")
        print(f"XML file: {self.xml_path}")
        print(f"Total tiles: {len(self.tiles)}")
        print(f"Total stitching pairs: {len(self.stitching_pairs)}")
        
        if self.tiles:
            sample_tile = next(iter(self.tiles.values()))
            print(f"\nTile specifications:")
            print(f"  Size: {sample_tile.size} voxels")
            print(f"  Voxel size: {sample_tile.voxel_size} {sample_tile.unit}")
            print(f"  Volume per tile: {sample_tile.size[0] * sample_tile.size[1] * sample_tile.size[2]:,} voxels")
        
        # Calculate grid layout
        if self.tiles:
            positions = [t.nominal_position for t in self.tiles.values() if t.nominal_position]
            if positions:
                x_positions = sorted(set(pos[0] for pos in positions))
                y_positions = sorted(set(pos[1] for pos in positions))
                z_positions = sorted(set(pos[2] for pos in positions))
                print(f"\nGrid layout:")
                print(f"  X positions: {len(x_positions)} unique ({min(x_positions):.1f} to {max(x_positions):.1f})")
                print(f"  Y positions: {len(y_positions)} unique ({min(y_positions):.1f} to {max(y_positions):.1f})")
                print(f"  Z positions: {len(z_positions)} unique ({min(z_positions):.1f} to {max(z_positions):.1f})")
        
        # Stitching quality
        if self.stitching_pairs:
            correlations = [p.correlation for p in self.stitching_pairs]
            print(f"\nStitching quality:")
            print(f"  Average correlation: {np.mean(correlations):.4f}")
            print(f"  Min correlation: {min(correlations):.4f}")
            print(f"  Max correlation: {max(correlations):.4f}")
            
        # Show some examples
        print(f"\nExample tile neighbors:")
        for i, (setup_id, tile) in enumerate(list(self.tiles.items())[:3]):
            neighbors = self.get_tile_neighbors(setup_id)
            neighbor_info = [(n[0], f"{n[1].correlation:.3f}") for n in neighbors]
            print(f"  Tile {setup_id}: {len(neighbors)} neighbors {neighbor_info}")
            
        # Add overlap analysis
        self.print_overlap_summary()

def main():
    parser = argparse.ArgumentParser(description='Analyze BigStitcher XML files')
    parser.add_argument('xml_file', help='Path to BigStitcher XML file')
    parser.add_argument('--output-tiles', help='Output CSV file for tile information')
    parser.add_argument('--output-stitching', help='Output CSV file for stitching results')
    parser.add_argument('--json', help='Output JSON file with complete analysis')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = BigStitcherAnalyzer(args.xml_file)
    
    # Parse data
    print("Parsing tiles...")
    tiles = analyzer.parse_tiles()

    print("Parsing stitching results...")
    stitching_pairs = analyzer.parse_stitching_results()
    
    # Print summary
    analyzer.print_summary()
    
    # Export data
    if args.output_tiles:
        analyzer.export_to_csv(args.output_tiles)
        
    if args.output_stitching:
        analyzer.export_stitching_to_csv(args.output_stitching)
        
    if args.json:
        # Convert to JSON-serializable format
        json_data = {
            'tiles': {},
            'stitching_pairs': []
        }
        
        for setup_id, tile in tiles.items():
            tile_dict = asdict(tile)
            # Convert numpy arrays to lists
            for transform in tile_dict['transforms']:
                if isinstance(transform['matrix'], np.ndarray):
                    transform['matrix'] = transform['matrix'].tolist()
            json_data['tiles'][str(setup_id)] = tile_dict
            
        for pair in stitching_pairs:
            pair_dict = asdict(pair)
            pair_dict['shift_matrix'] = pair_dict['shift_matrix'].tolist()
            json_data['stitching_pairs'].append(pair_dict)
            
        with open(args.json, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Complete analysis exported to {args.json}")

if __name__ == '__main__':
    main()
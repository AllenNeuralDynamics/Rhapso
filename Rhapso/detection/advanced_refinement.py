from collections import OrderedDict, defaultdict
import numpy as np
from scipy.spatial import cKDTree

"""
Final interest-point refinement:

1. Combine chunks by view.
2. Keep points inside overlap intervals.
3. Apply a spatially balanced point cap using refined DoG peak scores.
4. Suppress nearby duplicates strongest-first using DoG peak scores.
5. Preserve raw image intensity in the downstream output.
"""

class AdvancedRefinement:
    def __init__(self, interest_points, combine_distance, dataframes, overlapping_area, max_interval_size, max_spots):
        self.interest_points = interest_points
        self.combine_distance = float(combine_distance)
        self.image_loader_df = dataframes["image_loader"]
        self.overlapping_area = overlapping_area
        self.max_interval_size = float(max_interval_size)
        self.max_spots = int(max_spots)
        self.max_grid_bins_per_axis = 8
        self.consolidated_data = OrderedDict()
        self.consolidated_peak_scores = OrderedDict()
        self.stats = OrderedDict()

    @staticmethod
    def size(interval):
        lb, ub = (np.asarray(v, dtype=np.float64) for v in interval)
        return float(np.prod(np.maximum(ub - lb + 1.0, 0.0)))

    @staticmethod
    def _coerce(points, intensities, peak_scores):
        points = np.asarray(points, dtype=np.float32)

        if points.size == 0:
            empty_points = np.empty((0, 3), dtype=np.float32)
            empty_values = np.empty((0,), dtype=np.float32)
            return empty_points, empty_values, empty_values.copy()

        points = points.reshape(-1, 3)
        n = len(points)

        intensities = (
            np.zeros(n, dtype=np.float32)
            if intensities is None
            else np.asarray(intensities, dtype=np.float32).reshape(-1)
        )

        peak_scores = (
            intensities.copy()
            if peak_scores is None
            else np.asarray(peak_scores, dtype=np.float32).reshape(-1)
        )

        n = min(len(points), len(intensities), len(peak_scores))
        points, intensities, peak_scores = points[:n], intensities[:n], peak_scores[:n]

        finite = (
            np.all(np.isfinite(points), axis=1)
            & np.isfinite(intensities)
            & np.isfinite(peak_scores)
        )

        return points[finite], intensities[finite], peak_scores[finite]

    def _ordered_view_ids(self, available):
        available = set(available)
        ordered, seen = [], set()

        for _, row in self.image_loader_df.iterrows():
            try:
                view_id = f"timepoint: {int(row['timepoint'])}, setup: {int(row['view_setup'])}"
            except (TypeError, ValueError):
                continue

            if view_id in available and view_id not in seen:
                ordered.append(view_id)
                seen.add(view_id)

        ordered.extend(sorted(available - seen))
        return ordered

    def _collect_points_by_view(self):
        points_by_view = defaultdict(list)
        intensities_by_view = defaultdict(list)
        scores_by_view = defaultdict(list)
        intervals_by_view = defaultdict(list)

        for entry in self.interest_points:
            points, intensities, scores = self._coerce(
                entry.get("interest_points", []),
                entry.get("intensities"),
                entry.get("peak_scores"),
            )

            if len(points) == 0:
                continue

            view_id = entry["view_id"]
            points_by_view[view_id].append(points)
            intensities_by_view[view_id].append(intensities)
            scores_by_view[view_id].append(scores)
            intervals_by_view[view_id].append(entry["interval_key"])

        return points_by_view, intensities_by_view, scores_by_view, intervals_by_view
    
    @staticmethod
    def _contains(containing, contained):
        outer_lb, outer_ub = containing[:2]
        inner_lb, inner_ub = contained[:2]

        return all(
            outer_lb[d] <= inner_lb[d] and outer_ub[d] >= inner_ub[d]
            for d in range(3)
        )
    
    @staticmethod
    def _point_space_interval(points):
        return (
            np.min(points, axis=0).astype(np.float64),
            np.max(points, axis=0).astype(np.float64),
        )

    def _collect_overlaps_by_view(self):
        overlaps = defaultdict(list)

        for view_id, entries in self.overlapping_area.items():
            for entry in entries:
                lb = np.asarray(entry["lower_bound"], dtype=np.float64)
                ub = np.asarray(entry["upper_bound"], dtype=np.float64)

                if lb.shape != (3,) or ub.shape != (3,):
                    raise ValueError(f"Invalid overlap bounds for {view_id}: {lb.shape}, {ub.shape}")

                overlaps[view_id].append((lb, ub))

        return overlaps

    @staticmethod
    def _inside(points, interval):
        lb, ub = interval
        return np.all((points >= lb) & (points <= ub), axis=1)

    def _max_spots_for_interval(self, interval):
        if self.max_spots <= 0:
            return None
        if self.max_interval_size <= 0:
            return self.max_spots

        return max(1, int(round(self.max_spots * self.size(interval) / self.max_interval_size)))

    def _grid_shape(self, interval):
        lb, ub = interval
        span = np.maximum(ub - lb, 1.0)
        return np.clip(
            np.rint(self.max_grid_bins_per_axis * span / np.max(span)).astype(np.int64),
            1,
            self.max_grid_bins_per_axis,
        )

    def _spatial_top_n(self, points, intensities, scores, max_spots, interval):
        if max_spots is None or max_spots <= 0 or len(points) <= max_spots:
            return points, intensities, scores

        lb, ub = interval
        grid_shape = self._grid_shape(interval)
        normalized = (points.astype(np.float64) - lb) / np.maximum(ub - lb, 1.0)
        cells = np.floor(normalized * grid_shape).astype(np.int64)
        cells = np.clip(cells, 0, grid_shape - 1)

        by_cell = defaultdict(list)
        for i, cell in enumerate(cells):
            by_cell[tuple(int(v) for v in cell)].append(i)

        for indices in by_cell.values():
            indices.sort(key=lambda i: (-float(scores[i]), -float(intensities[i]), i))

        selected, depth = [], 0

        while len(selected) < max_spots:
            candidates = []

            for cell, indices in by_cell.items():
                if depth < len(indices):
                    i = indices[depth]
                    candidates.append((-float(scores[i]), -float(intensities[i]), cell, i))

            if not candidates:
                break

            candidates.sort()

            for _, _, _, i in candidates:
                selected.append(i)
                if len(selected) >= max_spots:
                    break

            depth += 1

        selected = np.asarray(selected, dtype=np.int64)
        return points[selected], intensities[selected], scores[selected]

    def filter(self):
        points_by_view, intensities_by_view, scores_by_view, intervals_by_view = self._collect_points_by_view()
        overlaps_by_view = self._collect_overlaps_by_view()

        selected_points = OrderedDict()
        selected_intensities = OrderedDict()
        selected_scores = OrderedDict()

        for view_id in self._ordered_view_ids(points_by_view):
            point_chunks = points_by_view[view_id]
            intensity_chunks = intensities_by_view[view_id]
            score_chunks = scores_by_view[view_id]
            chunk_intervals = intervals_by_view[view_id]
            overlap_intervals = overlaps_by_view.get(view_id, [])

            self.stats[view_id] = {
                "input_points": sum(len(points) for points in point_chunks),
                "overlap_selected_before_dedup": 0,
                "final_points": 0,
            }

            if not overlap_intervals:
                continue

            point_lists, intensity_lists, score_lists = [], [], []

            for overlap_interval in overlap_intervals:
                overlap_points = []
                overlap_intensities = []
                overlap_scores = []

                for points, intensities, scores, chunk_interval in zip(
                    point_chunks, intensity_chunks, score_chunks, chunk_intervals
                ):
                    # Both intervals are in detection/mipmap coordinates.
                    if not self._contains(overlap_interval, chunk_interval):
                        continue

                    overlap_points.append(points)
                    overlap_intensities.append(intensities)
                    overlap_scores.append(scores)

                if not overlap_points:
                    continue

                points = np.concatenate(overlap_points)
                intensities = np.concatenate(overlap_intensities)
                scores = np.concatenate(overlap_scores)

                points, intensities, scores = self._spatial_top_n(
                    points,
                    intensities,
                    scores,
                    self._max_spots_for_interval(overlap_interval),
                    self._point_space_interval(points),
                )

                point_lists.append(points)
                intensity_lists.append(intensities)
                score_lists.append(scores)

            if not point_lists:
                continue

            points = np.concatenate(point_lists)
            intensities = np.concatenate(intensity_lists)
            scores = np.concatenate(score_lists)

            selected_points[view_id] = points
            selected_intensities[view_id] = intensities
            selected_scores[view_id] = scores
            self.stats[view_id]["overlap_selected_before_dedup"] = len(points)

        return selected_points, selected_intensities, selected_scores

    def _suppress_duplicates(self, points, intensities, scores):
        if len(points) == 0:
            return points, intensities, scores
        if self.combine_distance < 0:
            raise ValueError("combine_distance must be non-negative")

        order = np.lexsort((np.arange(len(scores)), -intensities, -scores))
        tree = cKDTree(points)
        suppressed = np.zeros(len(points), dtype=bool)
        kept = []

        for i in order:
            if suppressed[i]:
                continue

            kept.append(i)
            nearby = tree.query_ball_point(points[i], r=self.combine_distance)
            suppressed[np.asarray(nearby, dtype=np.int64)] = True

        kept = np.asarray(kept, dtype=np.int64)
        return points[kept], intensities[kept], scores[kept]

    def kd_tree(self, points_by_view, intensities_by_view, scores_by_view):
        output, output_scores = OrderedDict(), OrderedDict()

        for view_id in self._ordered_view_ids(points_by_view):
            points, intensities, scores = self._suppress_duplicates(
                points_by_view[view_id],
                intensities_by_view[view_id],
                scores_by_view[view_id],
            )

            output[view_id] = [
                (point.tolist(), float(intensity))
                for point, intensity in zip(points, intensities)
            ]
            output_scores[view_id] = scores.astype(float).tolist()
            self.stats[view_id]["final_points"] = len(points)

        self.consolidated_data = output
        self.consolidated_peak_scores = output_scores

    def run(self):
        points_by_view, intensities_by_view, scores_by_view = self.filter()
        self.kd_tree(points_by_view, intensities_by_view, scores_by_view)

        print("Advanced Refinement Done")

        return self.consolidated_data
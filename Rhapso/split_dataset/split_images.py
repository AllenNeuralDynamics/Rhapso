from scipy.spatial import cKDTree
from copy import deepcopy
import time
import random
import math
import json
import fsspec
import numpy as np
import pandas as pd

class SplitImages:
    def __init__(self, target_image_size, target_overlap, min_step_size, data_gloabl, n5_path, point_density, min_points, 
                 max_points, error, excludeRadius):
        self.target_image_size = target_image_size
        self.target_overlap = target_overlap
        self.min_step_size = min_step_size
        self.data_global = data_gloabl

        self.image_loader_df = data_gloabl["image_loader"]
        self.view_setups_df = data_gloabl["view_setups"]
        self.view_registrations_df = data_gloabl["view_registrations"]
        self.view_interest_points_df = data_gloabl["view_interest_points"]

        self.alignment_path = str(n5_path).rstrip("/")
        self.point_density = point_density
        self.min_points = min_points
        self.max_points = max_points
        self.error = error
        self.exclude_radius = excludeRadius

        self.setup_definition = []
        self.point_manifest = {}

    def join_uri(self, *parts):
        return "/".join(
            str(part).strip("/") if i > 0 else str(part).rstrip("/")
            for i, part in enumerate(parts)
            if part is not None
        )

    def read_json(self, uri):
        fs, path = fsspec.core.url_to_fs(uri)

        with fs.open(path, "r") as f:
            return json.load(f)

    def read_parquet(self, uri):
        fs, path = fsspec.core.url_to_fs(uri)

        with fs.open(path, "rb") as f:
            return pd.read_parquet(f, engine="pyarrow")

    def load_point_manifest(self):
        manifest_uri = self.join_uri(self.alignment_path, "manifest.json")
        self.point_manifest = self.read_json(manifest_uri)

    def point_key(self, timepoint, setup, label):
        return f"{int(timepoint)}/{int(setup)}/{label}"
    
    def xml_point_path(self, timepoint, setup, label):
        return (
            f"tpId_{int(timepoint)}_"
            f"viewSetupId_{int(setup)}/"
            f"{label}"
        )

    def point_relative_path(self, timepoint, setup, label):
        return (
            f"points/"
            f"timepoint={int(timepoint)}/"
            f"setup={int(setup)}/"
            f"label={label}/"
            f"points.parquet"
        )

    def point_uri(self, timepoint, setup, label):
        rel_path = self.point_manifest["points"][self.point_key(timepoint, setup, label)]
        return self.join_uri(self.alignment_path, rel_path)

    def load_points_for_view_label(self, timepoint, setup, label):
        df = self.read_parquet(self.point_uri(timepoint, setup, label))

        if len(df) == 0:
            return np.empty((0, 3), dtype=np.float64)

        return df[["x", "y", "z"]].to_numpy(dtype=np.float64, copy=False)

    def intersect(self, interval, other_interval):
        n = len(interval[0])
        mins = [max(interval[0][d], other_interval[0][d]) for d in range(n)]
        maxs = [min(interval[1][d], other_interval[1][d]) for d in range(n)]

        return mins, maxs

    def create_models(self, transform_list):
        matrix = np.eye(4)

        for transform in transform_list:
            affine = np.fromstring(str(transform["affine"]).replace(",", " "), sep=" ").reshape(3, 4)
            matrix = matrix @ np.vstack([affine, [0, 0, 0, 1]])

        vals = matrix[:3, :].ravel()
        m00, m01, m02, m03, m10, m11, m12, m13, m20, m21, m22, m23 = map(float, vals)

        return {
            "type": "AffineTransform3D",
            "string": "3d-affine: (" + ", ".join(format(v, ".16g") for v in vals) + ")",
            "a": {
                "type": "AffineTransform3D$AffineMatrix3D",
                "m00": m00, "m01": m01, "m02": m02, "m03": m03,
                "m10": m10, "m11": m11, "m12": m12, "m13": m13,
                "m20": m20, "m21": m21, "m22": m22, "m23": m23,
                "m": [[m00, m01, m02, m03], [m10, m11, m12, m13], [m20, m21, m22, m23]],
            },
            "d0": {"type": "RealPoint", "string": f"({format(m00, '.16g')},{format(m10, '.16g')},{format(m20, '.16g')})", "n": 3, "position": [m00, m10, m20]},
            "d1": {"type": "RealPoint", "string": f"({format(m01, '.16g')},{format(m11, '.16g')},{format(m21, '.16g')})", "n": 3, "position": [m01, m11, m21]},
            "d2": {"type": "RealPoint", "string": f"({format(m02, '.16g')},{format(m12, '.16g')},{format(m22, '.16g')})", "n": 3, "position": [m02, m12, m22]},
            "ds": [[m00, m10, m20], [m01, m11, m21], [m02, m12, m22]],
        }

    def localizing_zero_min_interval_iterator(self, dimensions):
        dims = [int(d) for d in dimensions]
        n = len(dims)
        steps = [1] * n

        for d in range(1, n):
            steps[d] = steps[d - 1] * dims[d - 1]

        pos = [0] * n

        if n:
            pos[0] = -1

        return {
            "dimensions": dims,
            "index": -1,
            "last_index": (steps[-1] * dims[-1] - 1) if n else -1,
            "max": [d - 1 for d in dims],
            "min": [0] * n,
            "n": n,
            "position": pos,
            "steps": steps,
        }

    def split_dims(self, input_size, i, final_size, overlap):
        dim_intervals = []
        to_val = 0
        from_val = 0

        while to_val < input_size[i]:
            to_val = min(input_size[i], from_val + final_size - 1)
            dim_intervals.append((from_val, to_val))
            from_val = to_val - overlap + 1

        return dim_intervals

    def last_image_size(self, l, s, o):
        num = l - 2 * (s - o) - o
        den = s - o
        rem = num % den if num >= 0 else -((-num) % den)
        size = o + rem

        if size < 0:
            size = l + size

        return size

    def distribute_intervals_fixed_overlap(self, input_size):
        input_size = list(map(int, str(input_size).split()))

        for i in range(len(input_size)):
            if self.target_image_size[i] % self.min_step_size[i] != 0:
                raise RuntimeError(f"target size {self.target_image_size[i]} not divisible by min step size {self.min_step_size[i]} for dim {i}")

            if self.target_overlap[i] % self.min_step_size[i] != 0:
                raise RuntimeError(f"overlap {self.target_overlap[i]} not divisible by min step size {self.min_step_size[i]} for dim {i}")

        interval_basis = []

        for i, length in enumerate(input_size):
            if length <= self.target_image_size[i]:
                interval_basis.append([(0, length - 1)])
                continue

            s = self.target_image_size[i]
            o = self.target_overlap[i]
            last_size = self.last_image_size(length, s, o)

            if last_size == s:
                final_size = s
            elif last_size <= s // 2:
                final_size = s

                while True:
                    final_size += self.min_step_size[i]
                    next_last_size = self.last_image_size(length, final_size, o)
                    delta = last_size - next_last_size
                    last_size = next_last_size

                    if delta <= 0:
                        break
            else:
                final_size = s

                while True:
                    final_size -= self.min_step_size[i]
                    next_last_size = self.last_image_size(length, final_size, o)
                    delta = last_size - next_last_size
                    last_size = next_last_size

                    if delta >= 0:
                        break

                final_size += self.min_step_size[i]

            interval_basis.append(self.split_dims(input_size, i, final_size, self.target_overlap[i]))

        num_intervals = [len(x) for x in interval_basis]
        cursor = self.localizing_zero_min_interval_iterator(num_intervals)

        interval_list = []
        current_interval = [0, 0, 0]

        while cursor["index"] < cursor["last_index"]:
            cursor["index"] += 1

            for i in range(cursor["n"]):
                cursor["position"][i] += 1

                if cursor["position"][i] > cursor["max"][i]:
                    cursor["position"][i] = 0
                else:
                    break

            for i in range(cursor["n"]):
                current_interval[i] = cursor["position"][i]

            min_val = [0, 0, 0]
            max_val = [0, 0, 0]

            for i in range(len(input_size)):
                min_val[i], max_val[i] = interval_basis[i][current_interval[i]]

            interval_list.append((min_val, max_val))

        return interval_list

    def max_interval_spread(self, old_setups_df):
        max_val = 1

        for _, row in old_setups_df.iterrows():
            max_val = max(len(self.distribute_intervals_fixed_overlap(row["size"])), max_val)

        return max_val

    def is_empty(self, interval):
        mins, maxs = interval
        return any(mn > mx for mn, mx in zip(mins, maxs))

    def contains(self, ip, interval):
        for i in range(len(ip)):
            if ip[i] < interval[0][i] or ip[i] > interval[1][i]:
                return False

        return True

    def view_key(self, timepoint, setup):
        return f"timepoint: {int(timepoint)}, setup: {int(setup)}"

    def split_images(self, timepoints, interest_points, fake_label):
        old_setups_df = deepcopy(self.view_setups_df)
        old_registrations_df = deepcopy(self.view_registrations_df)

        new_interest_points = {}
        new_id = 0
        max_interval_spread = self.max_interval_spread(old_setups_df)
        rnd = random.Random(23424459)

        for _, row in old_setups_df.iterrows():
            old_id = int(row["id"])
            angle = row["angle"]
            channel = row["channel"]
            vox_dim = row["voxel_size"]
            vox_unit = row["voxel_unit"]
            illumination = row["illumination"]
            intervals = self.distribute_intervals_fixed_overlap(row["size"])
            interval_to_setup_id = {}
            local_new_tile_id = 0

            for i, interval in enumerate(intervals):
                new_setup_id = new_id
                new_tile_id = old_id * max_interval_spread + local_new_tile_id
                local_new_tile_id += 1

                interval_to_setup_id[(tuple(interval[0]), tuple(interval[1]))] = new_setup_id

                for t in timepoints:
                    t = int(t)
                    old_view_id = self.view_key(t, old_id)

                    old_vr = (
                        (old_registrations_df["timepoint"].astype(str) == str(t))
                        & (old_registrations_df["setup"].astype(str) == str(old_id))
                    )

                    transform_list = old_registrations_df.loc[old_vr, ["name", "type", "affine"]].to_dict("records")
                    mn, _ = interval

                    transform_list.append(
                        {
                            "name": "Image Splitting",
                            "affine": f"1, 0, 0, {mn[0]}, 0, 1, 0, {mn[1]}, 0, 0, 1, {mn[2]}",
                        }
                    )

                    new_view_id_key = self.view_key(t, new_setup_id)
                    new_v_ip_l = []

                    
                    if old_view_id in interest_points:
                        old_ip_l1 = interest_points[old_view_id]

                        split_points = []
                        point_id = 0

                        for ip in deepcopy(old_ip_l1["points"]):
                            if self.contains(ip, interval):
                                local_point = deepcopy(ip)

                                for d in range(len(interval[0])):
                                    local_point[d] -= interval[0][d]

                                split_points.append((point_id, local_point))
                                point_id += 1

                        split_label = "beads_split"
                        new_v_ip_l = [
                            {
                                "label": split_label,
                                "ip_list": {
                                    "interest_points": split_points,
                                    "point_path": self.xml_point_path(
                                        t,
                                        new_setup_id,
                                        split_label,
                                    ),
                                    "parameters": old_ip_l1["parameters_split"],
                                },
                            }
                        ]

                        fake_points = []
                        point_id = 0

                        for j in range(i):
                            other_interval = intervals[j]
                            intersection = self.intersect(interval, other_interval)

                            if self.is_empty(intersection):
                                continue

                            other_setup_id = interval_to_setup_id[(tuple(other_interval[0]), tuple(other_interval[1]))]
                            other_view_id = self.view_key(t, other_setup_id)
                            other_ip_list = new_interest_points[other_view_id]

                            n = len(interval[0])
                            num_pixels = 1

                            for k in range(n):
                                num_pixels *= intersection[1][k] - intersection[0][k] + 1

                            num_points = min(
                                self.max_points,
                                max(self.min_points, math.ceil(self.point_density * num_pixels / (100.0 * 100.0 * 100.0))),
                            )

                            other_points = next(x for x in other_ip_list if x["label"] == fake_label)["ip_list"]["interest_points"]
                            other_id = len(other_points)

                            search2 = None

                            if self.exclude_radius > 0 and len(other_points) > 0:
                                other_ip_global = []

                                for k, ip in enumerate(other_points):
                                    point_global = deepcopy(ip[1])

                                    for d in range(n):
                                        point_global[d] += other_interval[0][d]

                                    other_ip_global.append((k, point_global))

                                tree2 = cKDTree(np.vstack([point for _, point in other_ip_global]))

                                def search2(q_point_global, radius=self.exclude_radius):
                                    idxs = tree2.query_ball_point(np.asarray(q_point_global, float), radius)
                                    return [other_ip_global[k] for k in idxs]

                            tmp = [0.0] * n

                            for _ in range(num_points):
                                p = [0.0] * n
                                op = [0.0] * n

                                for d in range(n):
                                    l = rnd.random() * (intersection[1][d] - intersection[0][d] + 1) + intersection[0][d]

                                    p[d] = (l + (rnd.random() - 0.5) * self.error) - interval[0][d]
                                    op[d] = (l + (rnd.random() - 0.5) * self.error) - other_interval[0][d]
                                    tmp[d] = l

                                num_neighbors = 0

                                if search2 is not None:
                                    num_neighbors = len(search2(np.asarray(tmp, dtype=float), self.exclude_radius))

                                if num_neighbors == 0:
                                    fake_points.append((point_id, p))
                                    other_points.append((other_id, op))
                                    point_id += 1
                                    other_id += 1

                        new_v_ip_l.append(
                            {
                                "label": fake_label,
                                "ip_list": {
                                    "interest_points": fake_points,
                                    "point_path": self.xml_point_path(
                                        t,
                                        new_setup_id,
                                        fake_label,
                                    ),
                                    "parameters": old_ip_l1["parameters_fake"],
                                },
                            }
                        )

                    self.setup_definition.append(
                        {
                            "interval": interval,
                            "old_view": (t, old_id),
                            "new_view": (t, new_setup_id),
                            "voxel_dim": vox_dim,
                            "voxel_unit": vox_unit,
                            "angle": angle,
                            "channel": channel,
                            "illumination": illumination,
                            "old_models": transform_list,
                            "tile": new_tile_id,
                        }
                    )

                    # new_interest_points[new_view_id_key] = new_v_ip_l
                    
                    if new_v_ip_l:
                        new_interest_points[new_view_id_key] = new_v_ip_l

                new_id += 1

        return new_interest_points

    def load_interest_points(self):
        if self.view_interest_points_df.empty:
            return {}
        
        self.load_point_manifest()

        interest_points = {}

        for _, row in self.view_interest_points_df.iterrows():
            timepoint = int(row["timepoint"])
            setup = int(row["setup"])
            label = str(row["label"])
            view_id = self.view_key(timepoint, setup)

            overlap_px = f"[{self.target_overlap[0]}, {self.target_overlap[1]}, {self.target_overlap[2]}]"

            interest_points[view_id] = {
                "points": self.load_points_for_view_label(timepoint, setup, label),
                "base_path": self.alignment_path,
                "label": label,
                "parameters_split": str(row["params"]),
                "parameters_fake": (
                    f"Fake points for image splitting: overlapPx={overlap_px}, targetSize={self.target_image_size}, "
                    f"minStepSize={self.min_step_size}, optimize=true, pointDensity={self.point_density}, "
                    f"minPoints={self.min_points}, maxPoints={self.max_points}, error={self.error}, "
                    f"excludeRadius={self.exclude_radius}"
                ),
            }

        return interest_points

    def run(self):
        timepoints = sorted({int(row["timepoint"]) for _, row in self.image_loader_df.iterrows()})
        fake_label = f"splitPoints_{int(time.time() * 1000)}"

        interest_points = self.load_interest_points()
        new_split_interest_points = self.split_images(timepoints, interest_points, fake_label)
        print("Tiles Split")

        return new_split_interest_points, self.setup_definition
    
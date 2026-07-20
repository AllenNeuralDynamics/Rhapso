import json
from collections import defaultdict
import fsspec
import pandas as pd

class SaveMatches:
    """
    Save matched corresponding interest points into Parquet/JSON.
    """

    def __init__(self, all_results, n5_output_path, data_global, match_type):
        self.all_results = all_results
        self.alignment_output_prefix = str(n5_output_path).rstrip("/")
        self.data_global = data_global
        self.match_type = match_type
        self.match_index_rows = []
        self.manifest_matches = {}
        self.manifest_id_maps = {}

    def join_uri(self, *parts):
        return "/".join(
            str(part).strip("/") if i > 0 else str(part).rstrip("/")
            for i, part in enumerate(parts)
            if part is not None
        )

    def ensure_parent_dir(self, uri):
        fs, path = fsspec.core.url_to_fs(uri)
        parent = path.rsplit("/", 1)[0] if "/" in path else ""

        if parent:
            fs.makedirs(parent, exist_ok=True)

        return fs, path

    def write_json(self, uri, obj):
        fs, path = self.ensure_parent_dir(uri)

        with fs.open(path, "w") as f:
            json.dump(obj, f, indent=2, sort_keys=True)
            f.write("\n")

    def write_parquet(self, uri, df):
        fs, path = self.ensure_parent_dir(uri)

        with fs.open(path, "wb") as f:
            df.to_parquet(f, engine="pyarrow", index=False)

    def parse_view(self, view):
        view = str(view)
        timepoint = int(view.split("tpId=")[1].split(",")[0])
        setup = int(view.split("setupId=")[1].split(")")[0])

        return timepoint, setup

    def format_view(self, timepoint, setup):
        return f"(tpId={int(timepoint)}, setupId={int(setup)})"

    def view_label_key(self, timepoint, setup, label):
        return f"{int(timepoint)}/{int(setup)}/{label}"

    def target_key(self, timepoint, setup, label):
        return f"{int(timepoint)},{int(setup)},{label}"

    def correspondence_relative_path(self, timepoint, setup, label):
        return (
            f"matches/"
            f"timepoint={int(timepoint)}/"
            f"setup={int(setup)}/"
            f"label={label}/"
            f"correspondences.parquet"
        )

    def id_map_relative_path(self, timepoint, setup, label):
        return (
            f"matches/"
            f"timepoint={int(timepoint)}/"
            f"setup={int(setup)}/"
            f"label={label}/"
            f"id_map.json"
        )

    def labels_for_view(self, timepoint, setup):
        labels = self.data_global["viewsInterestPoints"][(timepoint, setup)]["label"]
        return labels if isinstance(labels, list) else [labels]

    def group_results(self):
        grouped = defaultdict(list)

        for idx_a, _, view_a, label_a, idx_b, _, view_b, label_b in self.all_results:
            tp_a, setup_a = self.parse_view(view_a)
            tp_b, setup_b = self.parse_view(view_b)

            grouped[(view_a, label_a)].append(
                {
                    "source_point_id": int(idx_a),
                    "target_point_id": int(idx_b),
                    "target_timepoint": int(tp_b),
                    "target_setup": int(setup_b),
                    "target_label": str(label_b),
                }
            )

            grouped[(view_b, label_b)].append(
                {
                    "source_point_id": int(idx_b),
                    "target_point_id": int(idx_a),
                    "target_timepoint": int(tp_a),
                    "target_setup": int(setup_a),
                    "target_label": str(label_a),
                }
            )

        return grouped

    def build_id_map(self, matches):
        target_keys = sorted(
            {
                self.target_key(
                    match["target_timepoint"],
                    match["target_setup"],
                    match["target_label"],
                )
                for match in matches
            }
        )

        return {key: i for i, key in enumerate(target_keys)}

    def correspondence_dataframe(self, matches, id_map):
        columns = {
            "source_point_id": pd.Series(dtype="uint64"),
            "target_point_id": pd.Series(dtype="uint64"),
            "target_view_id": pd.Series(dtype="uint32"),
            "target_timepoint": pd.Series(dtype="int32"),
            "target_setup": pd.Series(dtype="int32"),
            "target_label": pd.Series(dtype="string"),
        }

        if len(matches) == 0:
            return pd.DataFrame(columns)

        rows = []

        for match in matches:
            key = self.target_key(
                match["target_timepoint"],
                match["target_setup"],
                match["target_label"],
            )

            rows.append(
                {
                    "source_point_id": match["source_point_id"],
                    "target_point_id": match["target_point_id"],
                    "target_view_id": id_map[key],
                    "target_timepoint": match["target_timepoint"],
                    "target_setup": match["target_setup"],
                    "target_label": match["target_label"],
                }
            )

        df = pd.DataFrame(rows)

        df["source_point_id"] = df["source_point_id"].astype("uint64")
        df["target_point_id"] = df["target_point_id"].astype("uint64")
        df["target_view_id"] = df["target_view_id"].astype("uint32")
        df["target_timepoint"] = df["target_timepoint"].astype("int32")
        df["target_setup"] = df["target_setup"].astype("int32")
        df["target_label"] = df["target_label"].astype("string")

        return df

    def save_one_view_label(self, timepoint, setup, label, matches):
        id_map = self.build_id_map(matches)
        corr_df = self.correspondence_dataframe(matches, id_map)

        corr_rel_path = self.correspondence_relative_path(timepoint, setup, label)
        id_map_rel_path = self.id_map_relative_path(timepoint, setup, label)

        self.write_parquet(
            self.join_uri(self.alignment_output_prefix, corr_rel_path),
            corr_df,
        )

        self.write_json(
            self.join_uri(self.alignment_output_prefix, id_map_rel_path),
            id_map,
        )

        key = self.view_label_key(timepoint, setup, label)
        self.manifest_matches[key] = corr_rel_path
        self.manifest_id_maps[key] = id_map_rel_path

        self.match_index_rows.append(
            {
                "timepoint": int(timepoint),
                "setup": int(setup),
                "label": str(label),
                "correspondences_path": corr_rel_path,
                "id_map_json_path": id_map_rel_path,
                "num_correspondences": int(len(corr_df)),
                "num_target_views": int(len(id_map)),
            }
        )

    def save_correspondences(self):
        grouped = self.group_results()

        self.match_index_rows = []
        self.manifest_matches = {}
        self.manifest_id_maps = {}

        for timepoint, setup in self.data_global["viewsInterestPoints"].keys():
            view = self.format_view(timepoint, setup)

            for label in self.labels_for_view(timepoint, setup):
                matches = grouped.get((view, label), [])
                self.save_one_view_label(timepoint, setup, label, matches)

        index_df = pd.DataFrame(
            self.match_index_rows,
            columns=[
                "timepoint",
                "setup",
                "label",
                "correspondences_path",
                "id_map_json_path",
                "num_correspondences",
                "num_target_views",
            ],
        )

        index_df["timepoint"] = index_df["timepoint"].astype("int32")
        index_df["setup"] = index_df["setup"].astype("int32")
        index_df["label"] = index_df["label"].astype(str)
        index_df["correspondences_path"] = index_df["correspondences_path"].astype(str)
        index_df["id_map_json_path"] = index_df["id_map_json_path"].astype(str)
        index_df["num_correspondences"] = index_df["num_correspondences"].astype("int64")
        index_df["num_target_views"] = index_df["num_target_views"].astype("int64")

        self.write_parquet(
            self.join_uri(self.alignment_output_prefix, "matches", "match_index.parquet"),
            index_df,
        )

    def save_manifest(self):
        manifest = {
            "format": "rhapso-matches",
            "format_version": 1,
            "match_type": self.match_type,
            "storage": {
                "match_index": "matches/match_index.parquet",
                "matches_root": "matches/",
            },
            "correspondences": self.manifest_matches,
            "id_maps": self.manifest_id_maps,
        }

        self.write_json(
            self.join_uri(self.alignment_output_prefix, "matches", "manifest.json"),
            manifest,
        )

    def run(self):
        self.save_correspondences()
        self.save_manifest()
        print("Matches Saved")
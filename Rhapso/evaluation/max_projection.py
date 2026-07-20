import os
import re
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import boto3
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import s3fs
from matplotlib.lines import Line2D

class NominalMaxProjectionMosaic:
    def __init__(
        self,
        xml_path: str,
        out_dir: str,
        scale_level: str,
    ):
        self.xml_path = xml_path
        self.out_dir = out_dir
        self.scale_level = scale_level

    def run(self) -> None:
        self.ensure_dir(self.out_dir)
        root = self.load_xml_root(self.xml_path)
        setup_sizes = self.parse_view_setup_sizes(root)
        nominal_transforms = self.parse_named_transforms_from_xml(root)
        tile_records = self.parse_zarr_tile_paths_from_xml(root)
        tile_records = sorted(tile_records, key=lambda r: r["setup"])

        missing_transforms = [
            r["setup"] for r in tile_records if r["setup"] not in nominal_transforms
        ]
        if missing_transforms:
            raise RuntimeError(
                f"Missing transform 'Translation to Nominal Grid' for setups: {missing_transforms[:20]} "
                f"{'...' if len(missing_transforms) > 20 else ''}"
            )

        missing_sizes = [
            r["setup"] for r in tile_records if r["setup"] not in setup_sizes
        ]
        if missing_sizes:
            raise RuntimeError(
                f"Missing ViewSetup size for setups: {missing_sizes[:20]} "
                f"{'...' if len(missing_sizes) > 20 else ''}"
            )

        tile_proj_records = []

        for rec in tile_records:
            setup = rec["setup"]
            tile_path = rec["full_path"]
            nominal = nominal_transforms[setup]
            size_info = setup_sizes[setup]
            arr = self.open_ome_zarr_level(tile_path)

            if arr.ndim == 5:
                vol_zyx = arr[0, 0, :, :, :].astype(np.float32)
            elif arr.ndim == 3:
                vol_zyx = arr.astype(np.float32)
            else:
                raise RuntimeError(
                    f"Unexpected array shape for {tile_path}: {arr.shape}. "
                    "Expected 5D T,C,Z,Y,X or 3D Z,Y,X."
                )

            proj_yx = vol_zyx.max(axis=0).compute().astype(np.float32)

            tile_proj_records.append(
                {
                    "setup": setup,
                    "tile_x": rec["tile_x"],
                    "tile_y": rec["tile_y"],
                    "tile_z": rec["tile_z"],
                    "projection_raw": proj_yx,
                    "nominal_transform": nominal,
                    "full_size_x": size_info["size_x"],
                    "full_size_y": size_info["size_y"],
                    "full_size_z": size_info["size_z"],
                }
            )

        mosaic_png, centers = self.build_nominal_mosaic(
            tile_proj_records=tile_proj_records,
            output_name="max_projection.png",
        )

        rows = self.extract_pairwise_rows(root, xy_thresh_log2=2.0)

        self.draw_pairwise_links_on_mosaic(
            mosaic_png=mosaic_png,
            centers=centers,
            rows=rows,
            output_name="max_projection_with_links.png",
        )

    @staticmethod
    def ensure_dir(path: str) -> None:
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def load_xml_root(xml_path: str) -> ET.Element:
        if xml_path.startswith("s3://"):
            parsed = urlparse(xml_path)
            bucket = parsed.netloc
            key = parsed.path.lstrip("/")
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            return ET.fromstring(obj["Body"].read())
        return ET.parse(xml_path).getroot()

    @staticmethod
    def parse_tile_indices(tile_name: str):
        m = re.search(r"Tile_X_(\d+)_Y_(\d+)_Z_(\d+)", tile_name)
        if not m:
            raise RuntimeError(f"Could not parse tile indices from: {tile_name}")
        return int(m.group(1)), int(m.group(2)), int(m.group(3))

    @staticmethod
    def affine_12_to_4x4(affine_text: str) -> np.ndarray:
        vals = [float(v) for v in affine_text.split()]
        if len(vals) != 12:
            raise RuntimeError(f"Expected 12 affine values, got {len(vals)}: {affine_text}")
        mat = np.eye(4, dtype=np.float64)
        mat[0, 0:4] = vals[0:4]
        mat[1, 0:4] = vals[4:8]
        mat[2, 0:4] = vals[8:12]
        return mat

    @staticmethod
    def parse_affine_3x4(text: str) -> np.ndarray:
        vals = np.array(text.split(), dtype=float)
        if vals.size != 12:
            raise ValueError(f"Expected 12 values in 3x4 affine, got {vals.size}")
        return vals.reshape(3, 4)

    @staticmethod
    def is_pure_translation(aff: np.ndarray, atol: float = 1e-9) -> bool:
        return np.allclose(aff[:, :3], np.eye(3), atol=atol)

    @staticmethod
    def parse_view_setup_sizes(root: ET.Element):
        sizes = {}

        for vs in root.findall(".//ViewSetup"):
            setup_id_text = vs.findtext("id")
            size_text = vs.findtext("size")

            if setup_id_text is None or size_text is None:
                continue

            setup_id = int(setup_id_text)
            sx, sy, sz = [int(v) for v in size_text.split()]

            sizes[setup_id] = {
                "size_x": sx,
                "size_y": sy,
                "size_z": sz,
            }

        return sizes

    def parse_named_transforms_from_xml(self, root: ET.Element):
        transform_name = "Translation to Nominal Grid"
        transforms = {}

        for vr in root.findall(".//ViewRegistration"):
            setup = int(vr.get("setup"))
            tp = int(vr.get("timepoint", 0))

            if tp != 0:
                continue

            found = None

            for vt in vr.findall("ViewTransform"):
                name = vt.findtext("Name")
                affine_text = vt.findtext("affine")

                if name == transform_name and affine_text:
                    found = self.affine_12_to_4x4(affine_text)
                    break

            if found is not None:
                transforms[setup] = found

        return transforms

    def parse_zarr_tile_paths_from_xml(self, root: ET.Element):
        image_loader = root.find(".//ImageLoader")
        if image_loader is None:
            raise RuntimeError("No <ImageLoader> found in XML")

        zarr_base = image_loader.findtext("zarr")
        if not zarr_base:
            raise RuntimeError("No <zarr> base path found in XML ImageLoader")

        zarr_base = zarr_base.rstrip("/") + "/"
        tile_records = []

        for zg in image_loader.findall(".//zgroup"):
            rel_path = zg.get("path")
            if not rel_path:
                continue

            x_idx, y_idx, z_idx = self.parse_tile_indices(rel_path)

            tile_records.append(
                {
                    "setup": int(zg.get("setup")),
                    "tp": int(zg.get("tp", 0)),
                    "path": rel_path,
                    "full_path": zarr_base + rel_path,
                    "tile_x": x_idx,
                    "tile_y": y_idx,
                    "tile_z": z_idx,
                }
            )

        return tile_records

    def extract_pairwise_rows(self, root: ET.Element, xy_thresh_log2: float):
        sr = root.find(".//StitchingResults")
        if sr is None:
            return []

        rows = []
        seen = set()

        for pr in sr.findall("PairwiseResult"):
            a = int(pr.get("view_setup_a"))
            b = int(pr.get("view_setup_b"))

            shift_aff = self.parse_affine_3x4(pr.find("shift").text)
            if not self.is_pure_translation(shift_aff):
                raise RuntimeError(f"Non-translation detected between {a} and {b}")

            shifts = shift_aff[:, 3].astype(float)
            corr = float(pr.find("correlation").text)

            bb = np.array(
                pr.find("overlap_boundingbox").text.split(),
                dtype=float,
            ).reshape(2, 3)

            ext = bb[1] - bb[0]
            overlap_x, overlap_y, overlap_z = ext.tolist()

            eps = 1e-9
            x = max(abs(overlap_x), eps)
            y = max(abs(overlap_y), eps)

            if np.log2(x / y) > xy_thresh_log2:
                align = "top_bottom"
            elif np.log2(y / x) > xy_thresh_log2:
                align = "left_right"
            else:
                align = "corner"

            sx_round = round(float(shifts[0]), 3)
            sy_round = round(float(shifts[1]), 3)
            sz_round = round(float(shifts[2]), 3)
            corr_round = round(corr, 6)

            key = (min(a, b), max(a, b), sx_round, sy_round, sz_round, corr_round)
            if key in seen:
                continue

            seen.add(key)

            rows.append([
                a,
                b,
                sx_round,
                sy_round,
                sz_round,
                corr_round,
                float(overlap_x),
                float(overlap_y),
                float(overlap_z),
                align,
            ])

        return rows

    @staticmethod
    def contrast_stretch_percentile(
        arr: np.ndarray,
        ignore_zeros: bool = True,
    ) -> np.ndarray:
        black_percentile = 20
        white_percentile = 99

        arr = arr.astype(np.float32, copy=False)
        finite = np.isfinite(arr)

        if ignore_zeros:
            mask = finite & (arr > 0)
        else:
            mask = finite

        if not np.any(mask):
            return np.zeros_like(arr, dtype=np.float32)

        lo = float(np.percentile(arr[mask], black_percentile))
        hi = float(np.percentile(arr[mask], white_percentile))

        if hi <= lo:
            hi = float(np.nanmax(arr[mask]))
            lo = float(np.nanmin(arr[mask]))

        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)

        out = (arr - lo) / (hi - lo)
        out = np.clip(out, 0, 1)
        out[~finite] = 0
        return out.astype(np.float32)

    def open_ome_zarr_level(self, zarr_path: str):
        s3_anon = False

        if zarr_path.startswith("s3://"):
            s3 = s3fs.S3FileSystem(anon=s3_anon)
            store = s3fs.S3Map(root=zarr_path.rstrip("/"), s3=s3, check=False)
            return da.from_zarr(store, component=self.scale_level)

        return da.from_zarr(zarr_path, component=self.scale_level)

    def build_nominal_mosaic(
        self,
        tile_proj_records,
        output_name: str,
    ):
        if not tile_proj_records:
            return None, {}

        placed = []

        for rec in tile_proj_records:
            proj = rec["projection_raw"]
            tile_h, tile_w = proj.shape
            full_size_x = rec["full_size_x"]
            full_size_y = rec["full_size_y"]
            scale_x = full_size_x / tile_w
            scale_y = full_size_y / tile_h
            nominal = rec["nominal_transform"]
            tx_full = float(nominal[0, 3])
            ty_full = float(nominal[1, 3])
            x0 = int(round(tx_full / scale_x))
            y0 = int(round(ty_full / scale_y))

            placed.append(
                {
                    **rec,
                    "x0": x0,
                    "y0": y0,
                    "x1": x0 + tile_w,
                    "y1": y0 + tile_h,
                    "tile_w": tile_w,
                    "tile_h": tile_h,
                }
            )

        min_x = min(r["x0"] for r in placed)
        min_y = min(r["y0"] for r in placed)
        max_x = max(r["x1"] for r in placed)
        max_y = max(r["y1"] for r in placed)

        mosaic_w = max_x - min_x
        mosaic_h = max_y - min_y
        mosaic_raw = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)
        centers = {}

        for rec in placed:
            proj = rec["projection_raw"]
            y0 = rec["y0"] - min_y
            y1 = rec["y1"] - min_y
            x0 = rec["x0"] - min_x
            x1 = rec["x1"] - min_x
            region = mosaic_raw[y0:y1, x0:x1]
            np.maximum(region, proj, out=region)

            centers[rec["setup"]] = {
                "cx": x0 + 0.5 * rec["tile_w"],
                "cy": y0 + 0.5 * rec["tile_h"],
            }

        mosaic_png = self.contrast_stretch_percentile(
            mosaic_raw,
            ignore_zeros=True,
        )

        png_out_path = os.path.join(self.out_dir, output_name)
        plt.imsave(png_out_path, mosaic_png, cmap="gray", vmin=0, vmax=1)

        return mosaic_png, centers

    @staticmethod
    def edge_color(corr: float) -> str:
        if corr >= 0.90:
            return "tab:blue"
        elif corr >= 0.80:
            return "tab:green"
        elif corr >= 0.70:
            return "gold"
        return "red"

    def draw_pairwise_links_on_mosaic(
        self,
        mosaic_png: np.ndarray,
        centers,
        rows,
        output_name: str,
    ):
        if mosaic_png is None or not centers:
            return

        best_row_by_pair = {}

        for r in rows:
            a = int(r[0])
            b = int(r[1])
            corr = float(r[5])
            key = (min(a, b), max(a, b))

            if key not in best_row_by_pair or corr > float(best_row_by_pair[key][5]):
                best_row_by_pair[key] = r

        mosaic_h, mosaic_w = mosaic_png.shape
        fig_w = max(8.0, min(24.0, mosaic_w / 350.0))
        fig_h = max(8.0, min(24.0, mosaic_h / 350.0))

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        ax.imshow(mosaic_png, cmap="gray", vmin=0, vmax=1, origin="upper")

        for r in best_row_by_pair.values():
            a = int(r[0])
            b = int(r[1])
            corr = float(r[5])

            if a not in centers or b not in centers:
                continue

            ax.plot(
                [centers[a]["cx"], centers[b]["cx"]],
                [centers[a]["cy"], centers[b]["cy"]],
                color=self.edge_color(corr),
                linewidth=1.2,
                alpha=0.9,
                zorder=2,
            )

        n_tiles = len(centers)

        if n_tiles <= 30:
            marker_size = 160
            font_size = 8
        elif n_tiles <= 80:
            marker_size = 110
            font_size = 6
        else:
            marker_size = 70
            font_size = 5

        for setup, c in centers.items():
            ax.scatter(
                c["cx"],
                c["cy"],
                s=marker_size,
                color="white",
                edgecolors="black",
                marker="s",
                linewidths=1.0,
                zorder=5,
            )

            ax.text(
                c["cx"],
                c["cy"],
                str(setup),
                fontsize=font_size,
                color="black",
                ha="center",
                va="center",
                zorder=6,
            )

        legend_handles = [
            Line2D([0], [0], color="tab:blue", lw=2, label="corr ≥ 0.90"),
            Line2D([0], [0], color="tab:green", lw=2, label="0.80 ≤ corr < 0.90"),
            Line2D([0], [0], color="gold", lw=2, label="0.70 ≤ corr < 0.80"),
            Line2D([0], [0], color="red", lw=2, label="corr < 0.70"),
        ]

        fig.legend(
            handles=legend_handles,
            title="Link bands",
            loc="upper center",
            ncol=4,
            bbox_to_anchor=(0.5, 0.99),
            frameon=True,
        )

        ax.set_title("Max projection with pairwise links", pad=10)
        ax.set_axis_off()

        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])

        out_path = os.path.join(self.out_dir, output_name)
        fig.savefig(out_path, dpi=250, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)


if __name__ == "__main__":
    runner = NominalMaxProjectionMosaic(
        xml_path="s3://aind-open-data/HCR_831988-s1-ls2_2026-05-27_00-00-00_processed_2026-05-28_01-30-18/image_tile_alignment/bigstitcher.xml",
        out_dir="/Users/sean.fite/Desktop/max_projection_out",
        scale_level="2",
    )
    runner.run()
import xml.etree.ElementTree as ET
import numpy as np

"""
Get bounding box min/max AND per-view composed transforms + size
"""

class ComputeBBox():
    def __init__(self, aligned_xml_path, zarr_input_prefix):
        self.aligned_xml_path = aligned_xml_path
        self.zarr_input_prefix = zarr_input_prefix  

    def affine_3x4_to_4x4(self, aff_str: str) -> np.ndarray:
        vals = [float(x) for x in aff_str.strip().split()]
        if len(vals) != 12:
            raise ValueError(f"Expected 12 numbers in affine, got {len(vals)}: {aff_str[:80]}...")
        A = np.eye(4, dtype=np.float64)
        A[0, :4] = vals[0:4]
        A[1, :4] = vals[4:8]
        A[2, :4] = vals[8:12]
        return A

    def parse_view_setup_sizes(self, root: ET.Element) -> dict[int, tuple[int, int, int]]:
        sizes = {}
        vs = root.find("./SequenceDescription/ViewSetups")
        if vs is None:
            raise ValueError("Missing SequenceDescription/ViewSetups")

        for v in vs.findall("./ViewSetup"):
            sid = int(v.findtext("./id"))
            size_txt = v.findtext("./size")
            if size_txt is None:
                raise ValueError(f"Missing <size> for ViewSetup id={sid}")
            sx, sy, sz = (int(x) for x in size_txt.strip().split())
            sizes[sid] = (sx, sy, sz)  # X Y Z
        return sizes

    def parse_view_registration_affines(self, root: ET.Element) -> dict[tuple[int, int], list[np.ndarray]]:
        regs = {}
        vr = root.find("./ViewRegistrations")
        if vr is None:
            raise ValueError("Missing ViewRegistrations")

        for r in vr.findall("./ViewRegistration"):
            tp = int(r.attrib["timepoint"])
            setup = int(r.attrib["setup"])

            mats = []
            for vt in r.findall("./ViewTransform"):
                aff_txt = vt.findtext("./affine")
                if aff_txt is None:
                    continue
                mats.append(self.affine_3x4_to_4x4(aff_txt))

            regs[(tp, setup)] = mats if mats else [np.eye(4, dtype=np.float64)]
        return regs
    
    def parse_zgroup_paths(self, root: ET.Element) -> dict[tuple[int, int], str]:
        out = {}
        zgroups = root.find("./SequenceDescription/ImageLoader/zgroups")
        if zgroups is None:
            raise ValueError("Missing SequenceDescription/ImageLoader/zgroups")

        for zg in zgroups.findall("./zgroup"):
            setup = int(zg.attrib["setup"])
            tp = zg.attrib.get("timepoint", zg.attrib.get("tp"))
            tp = int(tp)

            # path can be either an attribute OR a child element
            p = zg.attrib.get("path") or zg.findtext("./path")
            if p is None:
                continue

            out[(tp, setup)] = p.strip()

        return out

    def compose_affines(self, mats: list[np.ndarray]) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        for M in mats:
            T = T @ M
        return T
    
    def sample_bbox_points_xyz(self, size_xyz: tuple[int, int, int]) -> np.ndarray:
        sx, sy, sz = size_xyz

        xs = [0.0, float(sx - 1)]
        ys = [0.0, float(sy - 1)]
        zs = [0.0, float(sz - 1)]

        xm = float(sx - 1) * 0.5
        ym = float(sy - 1) * 0.5
        zm = float(sz - 1) * 0.5

        pts = []

        # corners
        for x in xs:
            for y in ys:
                for z in zs:
                    pts.append([x, y, z, 1.0])

        # face centers
        pts += [
            [xm, ys[0], zm, 1.0],
            [xm, ys[1], zm, 1.0],
            [xs[0], ym, zm, 1.0],
            [xs[1], ym, zm, 1.0],
            [xm, ym, zs[0], 1.0],
            [xm, ym, zs[1], 1.0],
        ]

        # edge midpoints
        pts += [
            [xm, ys[0], zs[0], 1.0],
            [xm, ys[1], zs[1], 1.0],
            [xs[0], ym, zs[0], 1.0],
            [xs[1], ym, zs[1], 1.0],
            [xs[0], ys[0], zm, 1.0],
            [xs[1], ys[1], zm, 1.0],
        ]

        return np.array(pts, dtype=np.float64)

    def corners_xyz(self, size_xyz: tuple[int, int, int]) -> np.ndarray:
        sx, sy, sz = size_xyz
        xs = [0.0, float(sx - 1)]
        ys = [0.0, float(sy - 1)]
        zs = [0.0, float(sz - 1)]
        pts = np.array([[x, y, z, 1.0] for x in xs for y in ys for z in zs], dtype=np.float64)
        return pts

    def compute_global_bbox(self, xml_path: str):
        if xml_path.startswith("s3://"):
            import boto3
            from urllib.parse import urlparse
            u = urlparse(xml_path)
            body = boto3.client("s3").get_object(Bucket=u.netloc, Key=u.path.lstrip("/"))["Body"].read()
            root = ET.fromstring(body)
        else:
            root = ET.parse(xml_path).getroot()

        sizes = self.parse_view_setup_sizes(root)
        regs = self.parse_view_registration_affines(root)
        zpaths = self.parse_zgroup_paths(root)
        prefix = self.zarr_input_prefix
        
        gmin = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
        gmax = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

        view_registrations = {}  

        for (tp, setup), mats in regs.items():
            if setup not in sizes:
                raise ValueError(f"ViewRegistration refers to setup={setup}, but no matching ViewSetup size found.")

            T = self.compose_affines(mats)

            tile_rel = zpaths[(tp, setup)] 
            full_path = f"{prefix}/{tile_rel}" if prefix is not None else tile_rel

            view_registrations[(tp, setup)] = {
                "transform": T,
                "size": sizes[setup],
                "path": full_path, 
            }

            # pts = self.corners_xyz(sizes[setup])
            pts = self.sample_bbox_points_xyz(sizes[setup])
            w = (T @ pts.T).T[:, :3]

            gmin = np.minimum(gmin, w.min(axis=0))
            gmax = np.maximum(gmax, w.max(axis=0))

        bb_min = np.floor(gmin).astype(np.int64)
        bb_max = np.ceil(gmax).astype(np.int64)
        return bb_min, bb_max, view_registrations

    def run(self):
        return self.compute_global_bbox(self.aligned_xml_path)
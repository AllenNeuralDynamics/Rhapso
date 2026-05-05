import boto3
import json

class NeuroglancerLinkGeneratorZ1:
    def __init__(self, zarr_paths, json_upload_path, vmin, vmax,):
        self.zarr_paths= zarr_paths
        self.json_upload_path = json_upload_path
        self.vmin = vmin
        self.vmax = vmax

    def write_json_to_s3(self, s3_uri: str, payload: dict) -> None:
        if not s3_uri.startswith("s3://"):
            raise ValueError(f"Expected s3:// URI, got: {s3_uri}")
        bucket_and_key = s3_uri[5:]
        bucket, _, key = bucket_and_key.partition("/")
        if not bucket or not key:
            raise ValueError(f"Invalid S3 URI: {s3_uri}")

        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(payload, indent=4).encode("utf-8"),
            ContentType="application/json",
        )

    def parse_s3_path(s3_path):
        """
        Parse the S3 path to get the bucket name and the parent directory
        """
        if s3_path.startswith("s3://"):
            path_parts = s3_path[5:].split("/")
            bucket_name = path_parts[0]
            parent_directory = "/".join(path_parts[1:-1])  # Exclude the zarr file/directory itself
            return bucket_name, parent_directory
        else:
            raise ValueError("Invalid S3 path format")

    def generate_hcr_link(self) -> None:
        """
        Generate a single HCR Neuroglancer config that shows multiple fused Zarr
        datasets (e.g., channel_488 / channel_561 / channel_638) as separate layers.
        """
        # Default vox sizes
        vox_sizes = (9.201793828644069e-08, 9.201793828644069e-08, 4.4860451398192966e-07)

        # Create dimensions using extracted voxel sizes (array form to match example)
        dimensions = {
            "x": [vox_sizes[0], "m"],
            "y": [vox_sizes[1], "m"],
            "z": [vox_sizes[2], "m"],
            "c'": [1, ""],
            "t": [0.001, "s"],
        }

        # Pure identity transform, like your HCR_817076 example
        identity_transform = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        ]

        layers = []

        # Canonical per-channel default colors
        channel_color_map = {
            "405": "#690afe",  # violet
            "488": "#59d5f8",  # cyan / blue
            "561": "#f5b64a",  # yellow/orange
            "568": "#f5b64a",  # treat like 561
            "594": "#f28e2b",  # orange
            "638": "#f25b5b",  # red
            "640": "#f25b5b",  # red-ish
        }
        # Fallback palette if we don't recognize the channel
        fallback_colors = ["#59d5f8", "#f5b64a", "#f25b5b", "#690afe", "#7bd88a"]

        for idx, s3_path in enumerate(self.zarr_paths):
            # Extract channel from path (e.g., ".../channel_488.zarr" or ".../ch488/...")
            raw_channel = None
            if "channel_" in s3_path:
                raw_channel = s3_path.split("channel_")[1].split(".zarr")[0].split("/")[0]
            elif "/ch" in s3_path:
                # e.g. .../ch488/...
                tail = s3_path.split("/ch", 1)[1]
                raw_channel = tail.split("/", 1)[0]

            # Normalize to numbers if possible (strip non-digits)
            channel_digits = None
            if raw_channel is not None:
                channel_digits = "".join(c for c in raw_channel if c.isdigit())

            channel_key = channel_digits if channel_digits else None
            display_name = f"CH_{channel_key}" if channel_key else f"CH_{idx+1}"

            # Decide default color
            if channel_key in channel_color_map:
                default_color_hex = channel_color_map[channel_key]
            else:
                # Cycle fallback colors
                default_color_hex = fallback_colors[idx % len(fallback_colors)]

            source_array = [
                {
                    "url": f"zarr://{s3_path}",
                    "transform": {
                        "matrix": identity_transform,
                        "outputDimensions": {
                            "t": [0.001, "s"],
                            "c'": [1, ""],
                            "z": [vox_sizes[2], "m"],
                            "y": [vox_sizes[1], "m"],
                            "x": [vox_sizes[0], "m"],
                        },
                    },
                }
            ]

            # HCR-style shader with per-layer default color
            shader = (
                f"#uicontrol vec3 color color(default=\"{default_color_hex}\")\n"
                "#uicontrol invlerp normalized\n"
                "void main() {\n"
                "emitRGB(color * normalized());\n"
                "}"
            )

            layer = {
                "type": "image",
                "source": source_array,
                "localDimensions": {
                    "c'": [1, ""],
                },
                "shaderControls": {
                    "normalized": {
                        # 👇 this is what sets the sliders for every layer
                        "range": [self.vmin, self.vmax],
                    }
                },
                "shader": shader,
                "visible": True,
                "opacity": 1.0,
                "name": display_name,
                "blend": "additive",
            }

            layers.append(layer)

        config = {
            "dimensions": dimensions,
            "layers": layers,
            "showAxisLines": False,
            "showScaleBar": False,
        }

        # Build ng_link 
        ng_link = (
            "https://neuroglancer-demo.appspot.com/#!"
            f"{self.json_upload_path}"
        )

        final_output = {
            "ng_link": ng_link,
            **config,
        }

        self.write_json_to_s3(self.json_upload_path, final_output)
        print("✅ Uploaded multi-zarr HCR configuration to S3")
        print(f"   S3 JSON: {self.json_upload_path}")
        print(f"🔗 Neuroglancer Link: {ng_link}")
    
    def run(self):
        self.generate_hcr_link()

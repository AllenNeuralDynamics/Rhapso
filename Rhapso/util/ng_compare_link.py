#!/usr/bin/env python3

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import quote

import fsspec
import numpy as np
import zarr


BIGSTITCHER_PATH = "s3://aind-scratch-data/sean.fite/exaSPIM-qc-bigstitcher"
RHAPSO_PATH = "s3://aind-scratch-data/sean.fite/exaSPIM-qc-rhapso"

BIGSTITCHER_XML = "s3://aind-open-data/exaSPIM_791116_2026-06-09_16-31-33_processed_2026-06-16_12-28-34/tile_alignment/ip_split_affine_alignment/bigstitcher_affine.split.xml"
RHAPSO_XML = "s3://aind-scratch-data/sean.fite/exaSPIM-eval-2/rhapso-solver-split-affine.xml"

OUTPUT_PATH = "/Users/sean.fite/Desktop/results/ng.txt"
VIEWER = "https://neuroglancer-demo.appspot.com/#!"

AXIS_ORDER = ["x", "y", "z", "t", "c^"]

INTENSITY_MIN = 0.0
INTENSITY_MAX = 5000.0
OUTPUT_BRIGHTNESS = 0.35

BIGSTITCHER_VISIBLE = True
RHAPSO_VISIBLE = False

BIGSTITCHER_OPACITY = 1.0
RHAPSO_OPACITY = 1.0


def read_xml(path):
    if path.startswith("s3://"):
        anon = path.startswith("s3://aind-open-data/")
        with fsspec.open(path, "rb", anon=anon) as file:
            return ET.parse(file).getroot()

    return ET.parse(path).getroot()


def xml_text(element, name):
    for node in element.iter():
        if node.tag.rsplit("}", 1)[-1] == name:
            return node.text.strip() if node.text else None

    return None


def java_distinct_palette(count):
    palette = []

    for index in range(count):
        hue = np.float32(index) / np.float32(max(count, 1))
        hue = np.float32((hue - np.floor(hue)) * np.float32(6.0))
        fraction = np.float32(hue - np.floor(hue))
        p = np.float32(0.0)
        q = np.float32(1.0 - fraction)
        t = np.float32(fraction)
        sector = int(hue)

        if sector == 0:
            red, green, blue = np.float32(1.0), t, p
        elif sector == 1:
            red, green, blue = q, np.float32(1.0), p
        elif sector == 2:
            red, green, blue = p, np.float32(1.0), t
        elif sector == 3:
            red, green, blue = p, q, np.float32(1.0)
        elif sector == 4:
            red, green, blue = t, p, np.float32(1.0)
        else:
            red, green, blue = np.float32(1.0), p, q

        red = int(np.float32(red * np.float32(255.0) + np.float32(0.5)))
        green = int(np.float32(green * np.float32(255.0) + np.float32(0.5)))
        blue = int(np.float32(blue * np.float32(255.0) + np.float32(0.5)))
        palette.append((red << 16) | (green << 8) | blue)

    return palette


def get_origin_color_mapping(xml_path):
    root = read_xml(xml_path)
    tag = lambda element: element.tag.rsplit("}", 1)[-1]

    new_to_old = {}

    for element in root.iter():
        if tag(element) != "SetupIdDefinition":
            continue

        new_id = xml_text(element, "NewId")
        old_id = xml_text(element, "OldId")

        if new_id is not None and old_id is not None:
            new_to_old[int(new_id)] = int(old_id)

    if not new_to_old:
        raise ValueError(f"No SetupIdDefinition mappings found in {xml_path}")

    setup_illumination = {}

    for element in root.iter():
        if tag(element) != "ViewSetup":
            continue

        setup_id = xml_text(element, "id")
        if setup_id is None:
            continue

        setup_id = int(setup_id)
        illumination = xml_text(element, "illumination")
        setup_illumination[setup_id] = int(illumination) if illumination is not None else new_to_old.get(setup_id, setup_id)

    illumination_order = []
    origin_to_illumination = {}
    origin_counts = {}

    for new_id in sorted(new_to_old):
        old_id = new_to_old[new_id]
        illumination = setup_illumination.get(new_id, old_id)

        if illumination not in illumination_order:
            illumination_order.append(illumination)

        existing = origin_to_illumination.get(old_id)
        if existing is not None and existing != illumination:
            raise ValueError(f"Origin tile {old_id} maps to multiple illuminations: {existing} and {illumination}")

        origin_to_illumination[old_id] = illumination
        origin_counts[old_id] = origin_counts.get(old_id, 0) + 1

    origin_ids = sorted(origin_to_illumination)
    encoded_palette = java_distinct_palette(len(illumination_order))
    display_palette = java_distinct_palette(len(origin_ids))
    illumination_index = {illumination: index for index, illumination in enumerate(illumination_order)}

    mapping = []

    for display_index, old_id in enumerate(origin_ids):
        illumination = origin_to_illumination[old_id]
        encoded_index = illumination_index[illumination]

        mapping.append({
            "origin": old_id,
            "splits": origin_counts[old_id],
            "illumination": illumination,
            "encoded_index": encoded_index,
            "encoded_color": encoded_palette[encoded_index],
            "display_color": display_palette[display_index],
        })

    print(f"\n{Path(xml_path).name}: {len(origin_ids)} origin tiles")
    for item in mapping:
        print(
            f"  origin {item['origin']:2d} | splits {item['splits']:2d} | "
            f"illumination {item['illumination']:2d} | input #{item['encoded_color']:06X} "
            f"-> output #{item['display_color']:06X}"
        )

    return mapping


def build_qc_shader(xml_path):
    mapping = get_origin_color_mapping(xml_path)
    comparisons = []

    for item in mapping:
        encoded = item["encoded_color"]
        display = item["display_color"]

        input_red = ((encoded >> 16) & 255) / 255.0
        input_green = ((encoded >> 8) & 255) / 255.0
        input_blue = (encoded & 255) / 255.0

        output_red = ((display >> 16) & 255) / 255.0
        output_green = ((display >> 8) & 255) / 255.0
        output_blue = (display & 255) / 255.0

        comparisons.append(
            f"""
  distance = dot(chroma - vec3({input_red:.6f}, {input_green:.6f}, {input_blue:.6f}),
                 chroma - vec3({input_red:.6f}, {input_green:.6f}, {input_blue:.6f}));
  if (distance < bestDistance) {{
    bestDistance = distance;
    tileColor = vec3({output_red:.6f}, {output_green:.6f}, {output_blue:.6f});
  }}"""
        )

    return f"""
#uicontrol invlerp channel0(channel=0)
#uicontrol invlerp channel1(channel=1)
#uicontrol invlerp channel2(channel=2)

void main() {{
  float redSignal = channel0();
  float greenSignal = channel1();
  float blueSignal = channel2();
  float intensity = max(redSignal, max(greenSignal, blueSignal));

  if (intensity <= 0.000001) {{
    emitTransparent();
    return;
  }}

  vec3 chroma = vec3(redSignal, greenSignal, blueSignal) / intensity;
  float bestDistance = 1000.0;
  float distance = 0.0;
  vec3 tileColor = vec3(1.0);
{''.join(comparisons)}

  emitRGB(tileColor * intensity * {OUTPUT_BRIGHTNESS});
}}
""".strip()


def canonical_axis(axis, index, rank):
    name = str(axis.get("name", "")).lower()
    axis_type = str(axis.get("type", "")).lower()

    if name.startswith("x"):
        return "x"
    if name.startswith("y"):
        return "y"
    if name.startswith("z"):
        return "z"
    if name.startswith("t") or axis_type == "time":
        return "t"
    if name.startswith("c") or axis_type == "channel":
        return "c^"

    fallback = {
        5: ["t", "c^", "z", "y", "x"],
        4: ["c^", "z", "y", "x"],
        3: ["z", "y", "x"],
    }

    if rank not in fallback:
        raise ValueError(f"Unsupported dataset rank: {rank}")

    return fallback[rank][index]


def unit_factor(axis_name, unit):
    unit = str(unit or "").lower().strip()

    if axis_name in {"x", "y", "z"}:
        return {
            "m": 1.0,
            "meter": 1.0,
            "metre": 1.0,
            "mm": 1e-3,
            "millimeter": 1e-3,
            "millimetre": 1e-3,
            "um": 1e-6,
            "µm": 1e-6,
            "micrometer": 1e-6,
            "micrometre": 1e-6,
            "nm": 1e-9,
            "nanometer": 1e-9,
            "nanometre": 1e-9,
        }.get(unit, 1e-6)

    if axis_name == "t":
        return {
            "s": 1.0,
            "second": 1.0,
            "ms": 1e-3,
            "millisecond": 1e-3,
        }.get(unit, 1.0)

    return 1.0


def find_transform(transforms, transform_type, rank):
    field = "scale" if transform_type == "scale" else "translation"
    default = np.ones(rank, dtype=np.float64) if transform_type == "scale" else np.zeros(rank, dtype=np.float64)

    for transform in transforms:
        if transform.get("type") != transform_type:
            continue

        values = transform.get(field)
        if values is not None:
            return np.asarray(values, dtype=np.float64)

    return default


def load_dataset(path):
    path = path.rstrip("/")
    group = zarr.open_group(path, mode="r")
    multiscales = group.attrs.get("multiscales")

    if not multiscales:
        raise ValueError(f"{path} is missing multiscales metadata")

    metadata = multiscales[0]
    datasets = metadata.get("datasets")

    if not datasets:
        raise ValueError(f"{path} is missing multiscale datasets")

    dataset = datasets[0]
    array = group[dataset["path"]]
    rank = len(array.shape)
    raw_axes = metadata.get("axes") or [{} for _ in range(rank)]
    axes = [canonical_axis(axis, index, rank) for index, axis in enumerate(raw_axes)]

    transforms = dataset.get("coordinateTransformations", [])
    scale = find_transform(transforms, "scale", rank)
    translation = find_transform(transforms, "translation", rank)
    factors = np.asarray([
        unit_factor(axis_name, raw_axis.get("unit"))
        for axis_name, raw_axis in zip(axes, raw_axes)
    ], dtype=np.float64)

    return {
        "path": path,
        "dataset_path": dataset["path"],
        "shape": np.asarray(array.shape, dtype=np.float64),
        "axes": axes,
        "scale": scale,
        "translation": translation,
        "factors": factors,
    }


def validate_axes(dataset, dataset_name):
    missing_axes = [axis for axis in AXIS_ORDER if axis not in dataset["axes"]]

    if missing_axes:
        raise ValueError(f"{dataset_name} is missing axes {missing_axes}. Detected axes: {dataset['axes']}")


def build_dimensions(reference):
    dimensions = {}

    for axis_name in AXIS_ORDER:
        source_index = reference["axes"].index(axis_name)
        physical_scale = reference["scale"][source_index] * reference["factors"][source_index]
        unit = "m" if axis_name in {"x", "y", "z"} else "s" if axis_name == "t" else ""
        dimensions[axis_name] = [float(physical_scale), unit]

    return dimensions


def build_transform(dataset, dimensions):
    rank = len(dataset["axes"])
    matrix = [[0.0 for _ in range(rank + 1)] for _ in range(rank)]

    for output_index, axis_name in enumerate(AXIS_ORDER):
        input_index = dataset["axes"].index(axis_name)
        matrix[output_index][input_index] = 1.0

    return {"matrix": matrix, "outputDimensions": dimensions}


def dataset_bounds(dataset, dimensions):
    minimum = []
    maximum = []

    for axis_name in AXIS_ORDER:
        source_index = dataset["axes"].index(axis_name)
        scale = dataset["scale"][source_index]
        translation = dataset["translation"][source_index]
        factor = dataset["factors"][source_index]
        global_scale = dimensions[axis_name][0]

        physical_start = translation * factor
        physical_stop = physical_start + dataset["shape"][source_index] * scale * factor
        minimum.append(physical_start / global_scale)
        maximum.append(physical_stop / global_scale)

    return np.asarray(minimum), np.asarray(maximum)


def build_layer(name, dataset, xml_path, dimensions, visible, opacity):
    return {
        "type": "image",
        "name": name,
        "visible": visible,
        "opacity": opacity,
        "blend": "additive",
        "shader": build_qc_shader(xml_path),
        "shaderControls": {
            "channel0": {"range": [INTENSITY_MIN, INTENSITY_MAX]},
            "channel1": {"range": [INTENSITY_MIN, INTENSITY_MAX]},
            "channel2": {"range": [INTENSITY_MIN, INTENSITY_MAX]},
        },
        "source": [{
            "url": f"{dataset['path']}/|zarr2:",
            "transform": build_transform(dataset, dimensions),
        }],
    }


def build_state(bigstitcher, rhapso):
    validate_axes(bigstitcher, "BigStitcher")
    validate_axes(rhapso, "Rhapso")

    dimensions = build_dimensions(bigstitcher)
    big_min, big_max = dataset_bounds(bigstitcher, dimensions)
    rhapso_min, rhapso_max = dataset_bounds(rhapso, dimensions)

    minimum = np.minimum(big_min, rhapso_min)
    maximum = np.maximum(big_max, rhapso_max)
    center = (minimum + maximum) * 0.5
    extent = maximum - minimum

    x_extent = extent[AXIS_ORDER.index("x")]
    y_extent = extent[AXIS_ORDER.index("y")]
    z_extent = extent[AXIS_ORDER.index("z")]

    return {
        "dimensions": dimensions,
        "position": center.tolist(),
        "crossSectionScale": max(1.0, float(max(x_extent, y_extent) / 900.0)),
        "projectionScale": max(1024.0, float(max(x_extent, y_extent, z_extent) * 1.5)),
        "layers": [
            build_layer("BigStitcher", bigstitcher, BIGSTITCHER_XML, dimensions, BIGSTITCHER_VISIBLE, BIGSTITCHER_OPACITY),
            build_layer("Rhapso", rhapso, RHAPSO_XML, dimensions, RHAPSO_VISIBLE, RHAPSO_OPACITY),
        ],
        "selectedLayer": {"visible": True, "layer": "BigStitcher"},
        "layout": "xy",
        "showAxisLines": False,
        "showScaleBar": True,
        "crossSectionBackgroundColor": "#000000",
    }


def write_outputs(state):
    output_path = Path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    state_json = json.dumps(state, separators=(",", ":"))
    url = f"{VIEWER}{quote(state_json, safe='')}"

    output_path.write_text(url + "\n", encoding="utf-8")
    json_output_path = output_path.with_suffix(".json")
    json_output_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    print(f"\nWrote URL to:  {output_path}")
    print(f"Wrote JSON to: {json_output_path}")
    print(url)


def main():
    bigstitcher = load_dataset(BIGSTITCHER_PATH)
    rhapso = load_dataset(RHAPSO_PATH)

    print(f"BigStitcher axes: {bigstitcher['axes']}")
    print(f"Rhapso axes:      {rhapso['axes']}")

    state = build_state(bigstitcher, rhapso)
    write_outputs(state)


if __name__ == "__main__":
    main()
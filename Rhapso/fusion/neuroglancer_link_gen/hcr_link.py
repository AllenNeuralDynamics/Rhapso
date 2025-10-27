"""
Library for generating HCR (Hybridization Chain Reaction) neuroglancer links.
"""
import boto3
from typing import List, Optional
import json
from .ng_state import NgState
from botocore.exceptions import ClientError


def list_s3_zarr_folders(s3_path: str) -> List[str]:
    """
    List all .zarr folders in an S3 path.
    
    Parameters
    ----------
    s3_path : str
        S3 path in format s3://bucket/path/
        
    Returns
    -------
    List[str]
        List of .zarr folder names
    """
    # Parse S3 path
    if not s3_path.startswith("s3://"):
        raise ValueError("S3 path must start with s3://")
    
    path_parts = s3_path[5:].split("/", 1)
    bucket = path_parts[0]
    prefix = path_parts[1] if len(path_parts) > 1 and path_parts[1] else ""
    
    # Ensure prefix ends with / if it's not empty
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    
    s3_client = boto3.client('s3')
    zarr_folders = []
    
    try:
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
        
        for page in pages:
            if 'CommonPrefixes' in page:
                for prefix_info in page['CommonPrefixes']:
                    folder_name = prefix_info['Prefix'].rstrip('/').split('/')[-1]
                    if folder_name.endswith('.zarr'):
                        zarr_folders.append(folder_name)
    except ClientError as e:
        print(f"Error listing S3 objects: {e}")
        return []
    
    return sorted(zarr_folders)


def generate_hcr_link(
    s3_path: str,
    vmin: float = 90,
    vmax: float = 400,
    opacity: float = 1.0,
    blend: str = "additive",
    output_json_path: str = ".",
    dataset_name: Optional[str] = None,
    bucket_path: str = "aind-open-data",
) -> None:
    """
    Creates a neuroglancer link to visualize HCR dataset - handles both multi-channel 
    directories and single zarr files.

    Parameters
    ----------
    s3_path : str
        S3 path to either a directory containing .zarr folders or a single .zarr file
    vmin : float
        Minimum value for shader normalization range
    vmax : float
        Maximum value for shader normalization range
    opacity : float
        Opacity of layers
    blend : str
        Blend mode for layers
    output_json_path : str
        Local directory to write process_output.json file
    dataset_name : Optional[str]
        Name of dataset
    bucket_path : str
        S3 bucket name where the process_output.json will be uploaded
    """
    
    # Check if this is a single zarr file or a directory with multiple zarr folders
    if s3_path.endswith('.zarr') or '.zarr/' in s3_path:
        # Single zarr file - use ExaSPIM-like processing but with HCR formatting
        print("Processing single zarr file in HCR format...")
        return _generate_single_zarr_hcr_link(
            s3_path, vmin, vmax, opacity, blend, output_json_path, dataset_name, bucket_path
        )
    
    # Multi-channel HCR directory processing (original logic)
    zarr_folders = list_s3_zarr_folders(s3_path)
    
    if not zarr_folders:
        raise ValueError(f"No .zarr folders found in {s3_path}")
    
    print(f"Found {len(zarr_folders)} .zarr folders: {zarr_folders}")
    
    # Define the standard dimensions for multi-channel HCR data
    dimensions = {
        "x": {"voxel_size": 2.3371543469894166e-07, "unit": "meters"},
        "y": {"voxel_size": 2.3371543469894166e-07, "unit": "meters"}, 
        "z": {"voxel_size": 1e-06, "unit": "meters"},
        "c'": {"voxel_size": 1, "unit": ""},
        "t": {"voxel_size": 0.001, "unit": "seconds"}
    }
    
    # Define shader configuration in the format expected by NgLayer
    shader_config = {
        "color": "#690afe",
        "emitter": "RGB", 
        "vec": "vec3"
    }
    
    # Create layers for each zarr folder
    layers = []
    for zarr_folder in zarr_folders:
        # Extract channel name from folder name (e.g., "channel_405.zarr" -> "CH_405")
        if zarr_folder.startswith("channel_") and zarr_folder.endswith(".zarr"):
            channel_name = zarr_folder[8:-5]  # Remove "channel_" prefix and ".zarr" suffix
            display_name = f"CH_{channel_name}"
        else:
            # Fallback naming
            display_name = zarr_folder.replace(".zarr", "").upper()
        
        # Construct proper zarr source URL - always point to original data location
        # Clean the s3_path and ensure proper format
        clean_s3_path = s3_path.rstrip('/')
        if clean_s3_path.startswith("s3://"):
            # Use the s3_path as-is, just add zarr:// prefix and zarr_folder
            zarr_source = f"zarr://{clean_s3_path}/{zarr_folder}"
        else:
            # If no s3:// prefix, add it
            zarr_source = f"zarr://s3://{clean_s3_path}/{zarr_folder}"
        
        layer = {
            "type": "image",
            "source": zarr_source,
            "localDimensions": {
                "c'": {"voxel_size": 1, "unit": ""}
            },
            "shaderControls": {
                "normalized": {
                    "range": [vmin, vmax]
                }
            },
            "shader": shader_config,
            "visible": True,
            "opacity": opacity,
            "name": display_name,
            "blend": blend
        }
        layers.append(layer)
    
    # Create the neuroglancer configuration
    config = {
        "dimensions": dimensions,
        "layers": layers,
        "showAxisLines": False,
        "showScaleBar": False
    }
    
    # Generate the link using NgState
    # Always use "aind-open-data" for source paths, bucket_path is only for the JSON location
    neuroglancer_link = NgState(
        input_config=config,
        mount_service="s3",
        bucket_path="aind-open-data",  # Keep source paths pointing to original data
        output_dir=output_json_path,
        dataset_name=dataset_name,
    )
    
    neuroglancer_link.save_state_as_json()
    
    # Post-process the JSON to match the desired format
    _post_process_hcr_json(output_json_path, neuroglancer_link.json_name)
    
    print(neuroglancer_link.get_url_link())


def _post_process_hcr_json(output_dir: str, json_filename: str) -> None:
    """
    Post-process the generated JSON to match the HCR-specific format requirements.
    """
    import json
    from pathlib import Path
    
    json_path = Path(output_dir) / json_filename
    
    try:
        # Read the generated JSON
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Replace shader with the specific HCR shader string
        hcr_shader = "#uicontrol vec3 color color(default=\"#690afe\")\n#uicontrol invlerp normalized\nvoid main() {\nemitRGB(color * normalized());\n}"
        
        if "layers" in data:
            for layer in data["layers"]:
                if layer.get("type") == "image":
                    layer["shader"] = hcr_shader
                    
                    # Fix any corrupted source paths
                    source = layer.get("source", "")
                    if source:
                        # Clean up corrupted paths by removing duplicate prefixes and wrong buckets
                        cleaned_source = source
                        
                        # Remove any bucket references that shouldn't be in source paths
                        import re
                        
                        # Pattern to match and extract the correct zarr path
                        # This handles cases like "zarr://s3://wrong-bucket/zarr:/s3:/correct-path"
                        pattern = r'zarr://s3://[^/]+/zarr:/s3:/(.+)'
                        match = re.search(pattern, cleaned_source)
                        if match:
                            # Rebuild with correct format
                            correct_path = match.group(1)
                            cleaned_source = f"zarr://s3://{correct_path}"
                        else:
                            # Clean up other malformed patterns
                            cleaned_source = re.sub(r'zarr://s3://[^/]+/zarr://s3://', 'zarr://s3://', cleaned_source)
                            cleaned_source = re.sub(r'zarr:/s3:/', 'zarr://s3://', cleaned_source)
                        
                        layer["source"] = cleaned_source
        
        # Ensure showAxisLines and showScaleBar are False (as specified in the requirement)
        data["showAxisLines"] = False
        data["showScaleBar"] = False
        
        # Save the modified JSON back
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"✅ Post-processed JSON with HCR-specific formatting")
        
    except Exception as e:
        print(f"⚠️  Warning: Could not post-process JSON: {e}")


def _generate_single_zarr_hcr_link(
    s3_path: str,
    vmin: float,
    vmax: float, 
    opacity: float,
    blend: str,
    output_json_path: str,
    dataset_name: Optional[str],
    bucket_path: str
) -> None:
    """
    Generate HCR link for a single zarr file with ExaSPIM-like structure but HCR formatting.
    """
    from .parsers import OmeZarrParser
    import numpy as np
    from pathlib import Path
    
    # Get zarr metadata using existing parser
    try:
        # For single zarr files, use the path as-is for parsing
        base_zarr_path = s3_path
        if '.zarr/' in base_zarr_path:
            # If path includes resolution level, strip it for metadata parsing
            base_zarr_path = base_zarr_path.split('.zarr')[0] + '.zarr'
            
        vox_sizes = OmeZarrParser.extract_tile_vox_size(base_zarr_path)
        
        # Extract channel from filename (e.g., channel_488.zarr -> 488)
        channel_name = "488"  # default
        if "channel_" in s3_path:
            channel_part = s3_path.split("channel_")[1].split(".zarr")[0].split("/")[0]
            channel_name = channel_part
        
    except Exception as e:
        print(f"Warning: Could not extract zarr metadata: {e}")
        # Use fallback values from your example
        vox_sizes = (9.201793828644069e-08, 9.201793828644069e-08, 4.4860451398192966e-07)
        channel_name = "488"
    
    # Create dimensions using extracted voxel sizes
    dimensions = {
        "x": [vox_sizes[0], "m"],
        "y": [vox_sizes[1], "m"],
        "z": [vox_sizes[2], "m"],
        "c'": [1, ""],
        "t": [0.001, "s"]
    }
    
    # Create identity transform matrix (5x6 as per your example)
    identity_transform = [
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.1982632279396057],
        [0.0, 0.0, 0.0, 1.0, 0.0, -0.8770887851715088],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.507804274559021]
    ]
    
    # Create source array with transform
    source_array = [{
        "url": f"zarr://{s3_path}",
        "transform": {
            "matrix": identity_transform,
            "outputDimensions": {
                "t": [0.001, "s"],
                "c'": [1, ""],
                "z": [vox_sizes[2], "m"],
                "y": [vox_sizes[1], "m"],
                "x": [vox_sizes[0], "m"]
            }
        }
    }]
    
    # Create layer
    layer = {
        "type": "image",
        "source": source_array,
        "localDimensions": {
            "c'": [1, ""]
        },
        "shaderControls": {
            "normalized": {
                "range": [vmin, vmax]
            }
        },
        "shader": f"#uicontrol vec3 color color(default=\"#59d5f8\")\n#uicontrol invlerp normalized\nvoid main() {{\nemitRGB(color * normalized());\n}}",
        "visible": True,
        "opacity": opacity,
        "name": f"CH_{channel_name}",
        "blend": blend
    }
    
    # Create config
    config = {
        "dimensions": dimensions,
        "layers": [layer],
        "showAxisLines": False,
        "showScaleBar": False
    }
    
    # Generate output directly (don't use NgState to avoid path corruption)
    output_file = Path(output_json_path) / "process_output.json"
    
    # Add ng_link 
    if bucket_path != "aind-open-data":
        ng_link = f"https://neuroglancer-demo.appspot.com/#!s3://{bucket_path}/{dataset_name}/process_output.json"
    else:
        ng_link = f"https://neuroglancer-demo.appspot.com/#!s3://aind-open-data/{dataset_name}/process_output.json"
    
    final_output = {
        "ng_link": ng_link,
        **config
    }
    
    # Save JSON directly
    with open(output_file, 'w') as f:
        import json
        json.dump(final_output, f, indent=4)
    
    print(f"✅ Generated single zarr HCR configuration")
    print(f"🔗 Neuroglancer Link: {ng_link}")
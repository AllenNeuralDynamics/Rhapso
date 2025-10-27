import argparse
from pathlib import Path
import boto3
import json
import urllib.parse

from .exaspim_link import generate_exaspim_link
from .hcr_link import generate_hcr_link

def upload_to_s3(file_path, bucket_name, s3_file_path):
    """
    Upload a file to an S3 bucket

    :param file_path: File to upload
    :param bucket_name: Bucket to upload to
    :param s3_file_path: S3 object name
    """
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(file_path, bucket_name, s3_file_path)
        print(f"File {file_path} uploaded to {bucket_name}/{s3_file_path}")
    except Exception as e:
        print(f"Error uploading file: {e}")

def is_hcr_dataset(s3_path):
    """
    Determine if the S3 path contains HCR data by checking for multiple .zarr folders
    
    Parameters
    ----------
    s3_path : str
        S3 path to check
        
    Returns
    -------
    bool
        True if this appears to be an HCR dataset
    """
    try:
        from hcr_link import list_s3_zarr_folders
        zarr_folders = list_s3_zarr_folders(s3_path)
        # Consider it HCR if we find multiple .zarr folders or folders with "channel_" prefix
        return len(zarr_folders) > 1 or any(folder.startswith("channel_") for folder in zarr_folders)
    except Exception as e:
        print(f"Could not check for HCR dataset: {e}")
        return False

def parse_s3_path(s3_path):
    """
    Parse the S3 path to get the bucket name and the parent directory

    :param s3_path: S3 path (s3://bucket-name/path/to/zarr)
    :return: tuple (bucket_name, parent_directory)
    """
    if s3_path.startswith("s3://"):
        path_parts = s3_path[5:].split("/")
        bucket_name = path_parts[0]
        parent_directory = "/".join(path_parts[1:-1])  # Exclude the zarr file/directory itself
        return bucket_name, parent_directory
    else:
        raise ValueError("Invalid S3 path format")

def parse_s3_upload_path(json_upload_bucket, parent_directory, json_upload_path=None):
    """
    Construct the S3 upload path for the JSON file

    :param json_upload_bucket: S3 bucket name for uploading
    :param parent_directory: Parent directory from the zarr path
    :param json_upload_path: Optional exact S3 path to use
    :return: tuple (bucket_name, s3_key)
    """
    if json_upload_path:
        # Use the exact path specified by user
        s3_key = json_upload_path
    else:
        # Use auto-generated path based on zarr structure
        s3_key = f"{parent_directory}/process_output.json"
    return json_upload_bucket, s3_key


def generate_neuroglancer_link(
    zarr_path,
    vmin,
    vmax,
    json_upload_bucket=None,
    json_upload_path=None,
    json_local_output="results",
    dataset_type="auto",
    opacity=0.5,
    blend="default"
):
    """
    Generate Neuroglancer link and configuration files
    
    :param zarr_path: Path to the Zarr dataset (s3://bucket/path/to/zarr)
    :param vmin: Minimum value for scaling
    :param vmax: Maximum value for scaling
    :param json_upload_bucket: S3 bucket name to upload JSON
    :param json_upload_path: Exact S3 path within the bucket to upload JSON
    :param json_local_output: Local folder name to save the output JSON
    :param dataset_type: Processing type: 'auto', 'hcr', or 'exaspim'
    :param opacity: Opacity for the visualization
    :param blend: Blend mode
    """
    # Print input parameters
    print("=" * 60)
    print("NEUROGLANCER LINK GENERATOR")
    print("=" * 60)
    print(f"Input Parameters:")
    print(f"  Zarr Path: {zarr_path}")
    print(f"  VMin: {vmin}")
    print(f"  VMax: {vmax}")
    print(f"  Opacity: {opacity}")
    print(f"  Blend: {blend}")
    print(f"  Local Output Folder: {json_local_output}")
    print(f"  Upload Bucket: {json_upload_bucket if json_upload_bucket else 'None (using aind-open-data for link)'}")
    print(f"  Upload Path: {json_upload_path if json_upload_path else 'Auto-generated from zarr path'}")
    print("=" * 60)

    print("🔄 Processing...")

    # Determine the S3 bucket and parent directory
    s3_bucket, parent_directory = parse_s3_path(zarr_path)

    # Determine which bucket to use for the neuroglancer link
    bucket_path = json_upload_bucket if json_upload_bucket else "aind-open-data"

    # Create the local output directory path (relative to current working directory)
    local_output_path = Path.cwd() / json_local_output
    
    # Create the directory if it doesn't exist
    local_output_path.mkdir(parents=True, exist_ok=True)

    print("🔄 Generating Neuroglancer configuration...")

    # Decide processing mode: use CLI override when provided, otherwise auto-detect
    chosen_mode = dataset_type
    if chosen_mode == "auto":
        chosen_mode = "hcr" if is_hcr_dataset(zarr_path) else "exaspim"

    if chosen_mode == "hcr":
        print("📊 Using HCR processing...")
        generate_hcr_link(
            s3_path=zarr_path,
            vmin=vmin,
            vmax=vmax,
            opacity=1.0,  # Set opacity to 1.0 for HCR data as per spec
            blend="additive",
            output_json_path=str(local_output_path),
            dataset_name=parent_directory,
            bucket_path=bucket_path,
        )
    else:
        # ExaSPIM: ensure s3_path points at the .zarr root (trim trailing resolution index like '/0')
        normalized_path = zarr_path
        if ".zarr/" in normalized_path:
            # Keep everything up to and including '.zarr'
            normalized_path = normalized_path.split('.zarr')[0] + '.zarr'

        print("📊 Using ExaSPIM processing...")
        # Call the function with the parsed arguments
        generate_exaspim_link(
            None,
            s3_path=normalized_path,
            opacity=opacity,
            blend=blend,
            output_json_path=str(local_output_path),
            vmin=vmin,
            vmax=vmax,
            dataset_name=parent_directory,
            bucket_path=bucket_path,
        )

    # Define the local JSON file path
    output_json_file = local_output_path / "process_output.json"
    
    print("✅ Neuroglancer configuration generated!")
    
    # Read the generated JSON to get the ng_link
    ng_link_from_file = None
    neuroglancer_state = None
    try:
        with open(output_json_file, 'r') as f:
            json_content = json.load(f)
            ng_link_from_file = json_content.get("ng_link")
            # Remove the ng_link to get just the state for URL encoding
            neuroglancer_state = {k: v for k, v in json_content.items() if k != "ng_link"}
    except Exception as e:
        print(f"⚠️  Error reading generated JSON: {e}")
    
    print("🔄 Handling file operations...")
    
    # Handle S3 upload if json_upload_bucket is provided
    s3_upload_location = None
    if json_upload_bucket:
        try:
            upload_bucket, upload_key = parse_s3_upload_path(json_upload_bucket, parent_directory, json_upload_path)
            upload_to_s3(str(output_json_file), upload_bucket, upload_key)
            s3_upload_location = f"s3://{upload_bucket}/{upload_key}"
            print("✅ S3 upload completed!")

            # If a custom upload path was provided, update the ng_link inside the local JSON
            try:
                with open(output_json_file, 'r') as f:
                    current_json = json.load(f)

                # Construct the new ng_link that points to the uploaded S3 location
                new_ng_link = f"https://neuroglancer-demo.appspot.com/#!s3://{upload_bucket}/{'/'.join(upload_key.split('/')[:-1])}/{Path(output_json_file).name}" if '/' in upload_key else f"https://neuroglancer-demo.appspot.com/#!s3://{upload_bucket}/{Path(output_json_file).name}"

                # If user provided an explicit json_upload_path, prefer that exact key in the link
                if json_upload_path:
                    # If the json_upload_path includes a filename, use it; otherwise, append the default filename
                    key_parts = json_upload_path.split('/')
                    if key_parts[-1].endswith('.json'):
                        new_ng_link = f"https://neuroglancer-demo.appspot.com/#!s3://{upload_bucket}/{json_upload_path}"
                    else:
                        new_ng_link = f"https://neuroglancer-demo.appspot.com/#!s3://{upload_bucket}/{json_upload_path.rstrip('/')}/{Path(output_json_file).name}"

                # Update local JSON ng_link and write back
                current_json['ng_link'] = new_ng_link
                with open(output_json_file, 'w') as f:
                    json.dump(current_json, f, indent=2)

                # Update variable used for printing below
                ng_link_from_file = new_ng_link
            except Exception as e:
                print(f"⚠️  Warning: could not update local JSON ng_link to uploaded path: {e}")
        except Exception as e:
            print(f"❌ Error with S3 upload: {e}")
    
    # Generate results summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"📁 Local JSON saved: {output_json_file}")
    
    if s3_upload_location:
        print(f"☁️  S3 JSON uploaded: {s3_upload_location}")
    else:
        print("☁️  S3 Upload: None (no bucket specified)")
    
    print(f"🔗 Neuroglancer bucket: {bucket_path}")
    
    if ng_link_from_file:
        print(f"\n🌐 Neuroglancer Link (with JSON path):")
        print(f"   {ng_link_from_file}")
        
        # Create URL-encoded version
        if neuroglancer_state:
            try:
                state_json = json.dumps(neuroglancer_state, separators=(',', ':'))
                encoded_state = urllib.parse.quote(state_json)
                base_url = "https://neuroglancer-demo.appspot.com/"
                encoded_url = f"{base_url}#!{encoded_state}"
                
                # Save URL-encoded link to file
                encoded_url_file = local_output_path / "neuroglancer_encoded_url.txt"
                with open(encoded_url_file, 'w') as f:
                    f.write(encoded_url)
                
                print(f"\n💾 URL-encoded link saved to: {encoded_url_file}")
            except Exception as e:
                print(f"⚠️  Error creating URL-encoded link: {e}")
    else:
        print("⚠️  Could not extract Neuroglancer link from generated JSON")
    
    print("=" * 60)


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description="Generate Exaspim Link")

    # Add arguments
    parser.add_argument("--zarr_path", required=True, help="Path to the Zarr dataset (s3://bucket/path/to/zarr)")
    parser.add_argument("--opacity", type=float, default=0.5, help="Opacity for the visualization")
    parser.add_argument("--blend", default="default", help="Blend mode")
    parser.add_argument("--json_local_output", default="results", help="Local folder name to save the output JSON (e.g., 'results'). Will be created in current working directory.")
    parser.add_argument("--vmin", type=float, required=True, help="Minimum value for scaling")
    parser.add_argument("--vmax", type=float, required=True, help="Maximum value for scaling")
    parser.add_argument("--json_upload_bucket", default=None, help="S3 bucket name to upload JSON (e.g., martin-test-bucket). If not provided, no upload will occur and 'aind-open-data' will be used for the neuroglancer link.")
    parser.add_argument("--json_upload_path", default=None, help="Exact S3 path within the bucket to upload JSON (e.g., output/fuse/out.json). If not provided, will use auto-generated path based on zarr structure.")
    parser.add_argument("--dataset_type", choices=["auto","hcr","exaspim"], default="auto", help="Processing type: 'auto' detect HCR vs ExaSPIM, or force 'hcr' or 'exaspim'.")

    # Parse the arguments
    args = parser.parse_args()
    
    # Call the main function with parsed arguments
    generate_neuroglancer_link(
        zarr_path=args.zarr_path,
        vmin=args.vmin,
        vmax=args.vmax,
        json_upload_bucket=args.json_upload_bucket,
        json_upload_path=args.json_upload_path,
        json_local_output=args.json_local_output,
        dataset_type=args.dataset_type,
        opacity=args.opacity,
        blend=args.blend
    )

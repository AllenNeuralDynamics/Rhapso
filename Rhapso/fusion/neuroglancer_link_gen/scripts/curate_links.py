import argparse
import json
from typing import Any, Dict, List, Optional

import boto3

from ng_link.utils.utils import create_s3_client, list_folders_s3, save_to_csv


def read_process_output(
    s3_client: boto3.client, bucket_name: str, folder_name: str
) -> Optional[Dict[str, Any]]:
    """
    Read the process output JSON file from a specific folder in an S3 bucket.

    Parameters
    ----------
    s3_client : boto3.Client
        The S3 client object.
    bucket_name : str
        The name of the S3 bucket.
    folder_name : str
        The name of the folder.

    Returns
    -------
    dict or None
        A dictionary with the file content if successful, None otherwise.
    """
    file_key = f"{folder_name}/process_output.json"
    try:
        file_obj = s3_client.get_object(Bucket=bucket_name, Key=file_key)
        file_content = file_obj["Body"].read().decode("utf-8")
        return json.loads(file_content)
    except s3_client.exceptions.NoSuchKey:
        print(f"File not found: {file_key}")
    except Exception as e:
        print(f"Error reading file {file_key}: {e}")
        return None


def extract_ng_links(
    folders: List[str], s3_client: boto3.client, bucket_name: str
) -> List[Dict[str, str]]:
    """
    Extract 'ng_link' values from the process output JSON files in the
    specified folders.

    Parameters
    ----------
    folders : list
        A list of folder names.
    s3_client : boto3.Client
        The S3 client object.
    bucket_name : str
        The name of the S3 bucket.

    Returns
    -------
    list
        A list of dictionaries containing dataset names and
        their corresponding 'ng_link'.
    """
    results = []
    for folder in folders:
        json_content = read_process_output(s3_client, bucket_name, folder)
        if json_content:
            ng_link = json_content.get("ng_link")
            if ng_link:
                results.append({"Dataset Name": folder, "ng_link": ng_link})
    return results


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract 'ng_link' values from S3 and save to CSV."
    )
    parser.add_argument(
        "--bucket_name",
        type=str,
        required=True,
        help="The name of the S3 bucket.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        required=True,
        help="The prefix to filter folders.",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="The file path for the CSV file.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Main function to extract 'ng_link' values from S3 and save them to a CSV
    file.
    """
    args = parse_arguments()
    s3_client = create_s3_client()
    folders = list_folders_s3(s3_client, args.bucket_name, args.prefix)
    results = extract_ng_links(folders, s3_client, args.bucket_name)
    saved_file_path = save_to_csv(results, args.file_path)
    print(f"CSV file saved at {saved_file_path}")


# Example usage
if __name__ == "__main__":
    main()

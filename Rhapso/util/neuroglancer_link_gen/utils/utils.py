"""
Utility functions
"""
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Union

import boto3
import pandas as pd

# IO types
PathLike = Union[str, Path]


def create_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Create new folders.

    Parameters
    ------------------------
    dest_dir: PathLike
        Path where the folder will be created if it does not exist.
    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------
    OSError:
        if the folder exists.

    """

    if not (os.path.exists(dest_dir)):
        try:
            if verbose:
                print(f"Creating new directory: {dest_dir}")
            os.makedirs(dest_dir)
        except OSError as e:
            if e.errno != os.errno.EEXIST:
                raise


def delete_folder(dest_dir: PathLike, verbose: Optional[bool] = False) -> None:
    """
    Delete a folder path.
    Parameters
    ------------------------
    dest_dir: PathLike
        Path that will be removed.
    verbose: Optional[bool]
        If we want to show information about the folder status. Default False.

    Raises
    ------------------------
    shutil.Error:
        If the folder could not be removed.

    Returns
    ------------------------
        None

    """
    if os.path.exists(dest_dir):
        try:
            shutil.rmtree(dest_dir)
            if verbose:
                print(f"Folder {dest_dir} was removed!")
        except shutil.Error as e:
            print(f"Folder could not be removed! Error {e}")


def execute_command_helper(
    command: str,
    print_command: bool = False,
    stdout_log_file: Optional[PathLike] = None,
) -> None:
    """
    Execute a shell command.

    Parameters
    ------------------------
    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------
    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    if print_command:
        print(command)

    if stdout_log_file and len(str(stdout_log_file)):
        save_string_to_txt("$ " + command, stdout_log_file, "a")

    popen = subprocess.Popen(
        command, stdout=subprocess.PIPE, universal_newlines=True, shell=True
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield str(stdout_line).strip()
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def execute_command(config: dict) -> None:
    """
    Execute a shell command with a given configuration.

    Parameters
    ------------------------
    command: str
        Command that we want to execute.
    print_command: bool
        Bool that dictates if we print the command in the console.

    Raises
    ------------------------
    CalledProcessError:
        if the command could not be executed (Returned non-zero status).

    """

    for out in execute_command_helper(
        config["command"], config["verbose"], config["stdout_log_file"]
    ):
        if len(out):
            config["logger"].info(out)

        if config["exists_stdout"]:
            save_string_to_txt(out, config["stdout_log_file"], "a")


def check_path_instance(obj: object) -> bool:
    """
    Checks if an objects belongs to pathlib.Path subclasses.

    Parameters
    ------------------------
    obj: object
        Object that wants to be validated.

    Returns
    ------------------------
    bool:
        True if the object is an instance of Path subclass, False otherwise.
    """

    for childclass in Path.__subclasses__():
        if isinstance(obj, childclass):
            return True

    return False


def save_dict_as_json(
    filename: str, dictionary: dict, verbose: Optional[bool] = False
) -> None:
    """
    Saves a dictionary as a json file.

    Parameters
    ------------------------
    filename: str
        Name of the json file.
    dictionary: dict
        Dictionary that will be saved as json.
    verbose: Optional[bool]
        True if you want to print the path where the file was saved.

    """

    if dictionary is None:
        dictionary = {}

    else:
        for key, value in dictionary.items():
            # Converting path to str to dump dictionary into json
            if check_path_instance(value):
                # TODO fix the \\ encode problem in dump
                dictionary[key] = str(value)

    with open(filename, "w") as json_file:
        json.dump(dictionary, json_file, indent=4)

    if verbose:
        print(f"- Json file saved: {filename}")


def read_json_as_dict(filepath: str) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------
    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------
    dict:
        Dictionary with the data the json has.

    """

    dictionary = None

    if os.path.exists(filepath):
        with open(filepath) as json_file:
            dictionary = json.load(json_file)

    return dictionary


def save_string_to_txt(txt: str, filepath: PathLike, mode="w") -> None:
    """
    Saves a text in a file in the given mode.

    Parameters
    ------------------------
    txt: str
        String to be saved.

    filepath: PathLike
        Path where the file is located or will be saved.

    mode: str
        File open mode.

    """

    with open(filepath, mode) as file:
        file.write(txt + "\n")


def create_s3_client() -> boto3.client:
    """
    Create and return a boto3 S3 client.

    Returns
    -------
    boto3.Client
        A boto3 S3 client object.
    """
    return boto3.client("s3")


def list_folders_s3(
    s3_client: boto3.client, bucket_name: str, prefix: str
) -> list:
    """
    List top-level folders in an S3 bucket with a specified prefix.

    Parameters
    ----------
    s3_client : boto3.Client
        The S3 client object.
    bucket_name : str
        The name of the S3 bucket.
    prefix : str
        The prefix to filter folders.

    Returns
    -------
    list
        A list of folder names.
    """
    response = s3_client.list_objects_v2(
        Bucket=bucket_name, Prefix=prefix, Delimiter="/"
    )
    return [
        content.get("Prefix").rstrip("/")
        for content in response.get("CommonPrefixes", [])
    ]


def save_to_csv(data: List[dict], file_path: str) -> str:
    """
    Save the given data to a CSV file.

    Parameters
    ----------
    data : List[dict]
        The data to be saved.
    file_path : str
        The file path for the CSV file.

    Returns
    -------
    str
        The path of the saved CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    return file_path

"""
Utility functions for pipelines
"""


def fetch_local_xml(file_path):
    """
    Read XML content from a local file.

    Parameters
    ----------
    file_path : str
        Path to the XML file

    Returns
    -------
    str
        XML file contents
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find XML file at '{file_path}'")
    except Exception as e:
        raise RuntimeError(f"Error reading XML file at '{file_path}': {e}")

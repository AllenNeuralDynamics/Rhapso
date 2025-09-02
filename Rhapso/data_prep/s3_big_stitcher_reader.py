import json
import os
import s3fs

"""
Utility class for downloading BigStitcher outputs from S3 to local storage for N5 reader compatibility
"""

class S3BigStitcherReader:
    def __init__(self, s3_uri, local_directory):
           self.s3_uri = s3_uri
           self.local_directory = local_directory
    
    def download_n5_from_s3_to_local(self):
        """
        Recursively download an N5 dataset from S3 to a local directory.
        """
        s3 = s3fs.S3FileSystem(anon=False)
        s3_path = self.s3_uri.replace("s3://", "")
        all_keys = s3.find(s3_path, detail=True)

        for key, obj in all_keys.items():
            if obj["type"] == "file":
                rel_path = key.replace(s3_path + "/", "")
                local_file_path = os.path.join(self.local_directory, rel_path)
                os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
                s3.get(key, local_file_path)

                # Check for the specific interestpoints path
                if rel_path.endswith("beads/interestpoints/attributes.json") and "interestpoints.n5" in rel_path:
                    # Construct the path to the attributes file
                    attributes_path = os.path.join(os.path.dirname(local_file_path), "attributes.json")
                    attributes_data = {
                        "pointcloud": "1.0.0",
                        "type": "list",
                        "list version": "1.0.0"
                    }

                    with open(attributes_path, "w") as f:
                                        json.dump(attributes_data, f, indent=2)

    def run(self):
        self.download_n5_from_s3_to_local()

        s3_path = self.s3_uri.replace("s3://", "")
        full_local_path = os.path.join(self.local_directory, s3_path)

        # Final paths
        xml_input_path = os.path.join(full_local_path, "bigstitcher_ip.xml")
        n5_output_path = os.path.join(full_local_path, "interestpoints.n5")

        print("XML Input Path:", xml_input_path)
        print("N5 Output Path:", n5_output_path)

        return xml_input_path, n5_output_path
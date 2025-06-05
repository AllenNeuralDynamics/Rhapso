# Fetch helper functions


def fetch_local_xml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def fetch_from_s3(s3, bucket_name, input_file):
    response = s3.get_object(Bucket=bucket_name, Key=input_file)
    return response["Body"].read().decode("utf-8")

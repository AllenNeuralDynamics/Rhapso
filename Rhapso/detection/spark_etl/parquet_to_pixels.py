import boto3
import s3fs  
import pyarrow.parquet as pq
import matplotlib.pyplot as plt

def list_s3_bucket_contents(s3_client, bucket_name, prefix=''):
    paginator = s3_client.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    objects_info = []
    for page in page_iterator:
        if 'Contents' in page:
            for obj in page['Contents']:
                obj_key = obj['Key']
                objects_info.append({'key': obj_key})

    return objects_info

def read_parquet_from_s3(s3_uri):
    fs = s3fs.S3FileSystem()
    table = pq.read_table(s3_uri, filesystem=fs) 
    df = table.to_pandas()
    return df

def reshape_data(df, dimensions):
    dimensions_tuple = tuple(map(int, dimensions.split('x')))
    flat_array = df.to_numpy()
    image_chunk = flat_array.reshape(dimensions_tuple)
    print(image_chunk)
    return image_chunk

def visualize_slice(image):
        try:
            plt.figure(figsize=(10, 8))
            plt.imshow(image, cmap='gray')  
            plt.title(f"Image Slice")
            plt.colorbar()
            plt.show()
        except Exception as e:
            print(f"Error displaying slice: {e}")

def visualize_slice(image):
    z_level = 8
    try:
        slice_image = image[z_level, :, :]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(slice_image, cmap='gray')  
        plt.title(f"Image Slice at Z={z_level}")
        plt.colorbar()
        plt.show()
    except Exception as e:
        print(f"Error displaying slice: {e}")

def main():
    s3_client = boto3.client('s3')
    bucket_name = 'interest-point-detection'
    prefix = 'flattened-images-2'
    bucket_contents = list_s3_bucket_contents(s3_client, bucket_name, prefix)

    for path_key in bucket_contents:
        key = path_key['key']
        parts = key.split('/')
        filename = parts[-2]  
        details = filename.split('_')
       
        timepoint = details[1]  
        setup = details[3]  
        dimensions = details[4]   
        s3_uri = 's3://interest-point-detection/' + key

        df = read_parquet_from_s3(s3_uri)
        image = reshape_data(df, dimensions)
        print(key)
        visualize_slice(image)
        
    print("hi")

if __name__ == '__main__':
    main()

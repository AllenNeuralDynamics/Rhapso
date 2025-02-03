import Rhapso

# Call the xmlToDataframe function
# myDataframe = Rhapso.xmlToDataframe("/mnt/c/Users/marti/Documents/Allen/repos/Rhapso-Sample-Data/IP_TIFF_XML/dataset.xml")
myDataframe = Rhapso.xmlToDataframe("s3://rhapso-dev/rhapso-sample-data/dataset.xml")
print('myDataframe = ', myDataframe)

# Call the runOverlapDetection function

overlapDetection = Rhapso.OverlapDetection()

output = overlapDetection.run(myDataframe)

print("Overlap Detection Output: ", output)

# Example of sending the output to a .txt file in S3 location
# s3 = boto3.client('s3')
# bucket_name = 'your-bucket-name'
# output_file = 'path/to/output/file.txt'
# buffer = io.BytesIO()
# buffer.write(str(output).encode('utf-8'))
# buffer.seek(0)
# s3.put_object(Bucket=bucket_name, Body=buffer, Key=output_file)
# print(f"Sent to S3: {output_file} in bucket: {bucket_name}")
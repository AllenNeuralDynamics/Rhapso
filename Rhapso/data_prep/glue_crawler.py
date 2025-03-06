import boto3
import time

class GlueCrawler:
    def __init__(self, crawler_name, crawler_s3_path, crawler_database_name, crawler_iam_role):
        self.crawler_name = crawler_name
        self.s3_path = crawler_s3_path
        self.database_name = crawler_database_name
        self.iam_role = crawler_iam_role
        self.glue_client = boto3.client('glue')

    def create_or_update_crawler(self):
        crawlers = self.glue_client.list_crawlers()['CrawlerNames']
        if self.crawler_name in crawlers:
            print(f"Crawler '{self.crawler_name}' already exists.")
        else:
            try:
                response = self.glue_client.create_crawler(
                    Name=self.crawler_name,
                    Role=self.iam_role,
                    DatabaseName=self.database_name,
                    Description="Crawler for image data in Parquet format",
                    Targets={'S3Targets': [{'Path': self.s3_path}]},
                    SchemaChangePolicy={
                        'UpdateBehavior': 'UPDATE_IN_DATABASE',
                        'DeleteBehavior': 'DEPRECATE_IN_DATABASE'
                    }
                )
                print(f"Crawler '{self.crawler_name}' created successfully.")
            except Exception as e:
                print(f"Failed to create crawler: {str(e)}")

    def start_crawler(self):
        try:
            self.glue_client.start_crawler(Name=self.crawler_name)
            print(f"Crawler '{self.crawler_name}' started.")
        except Exception as e:
            print(f"Failed to start crawler: {str(e)}")

    def wait_for_crawler(self):
        timeout = 800  # Maximum wait time in seconds
        start_time = time.time()
        
        while True:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print("Timed out waiting for crawler to finish.")
                break

            crawler_state = self.glue_client.get_crawler(Name=self.crawler_name)['Crawler']['State']
            print(f"Waiting for crawler '{self.crawler_name}'. Current state: {crawler_state}")
            
            if crawler_state == 'READY':
                print(f"Crawler '{self.crawler_name}' is now ready for new operations.")
                break
            elif crawler_state == 'FAILED':
                print(f"Crawler '{self.crawler_name}' failed. Please check AWS Glue Console for details.")
                break
            elif crawler_state == 'STOPPING':
                print(f"Crawler '{self.crawler_name}' is stopping. Waiting until it fully stops...")
                time.sleep(15)
            else:
                time.sleep(15)

    def run(self):
        self.create_or_update_crawler()
        self.start_crawler()
        self.wait_for_crawler()
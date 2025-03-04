import boto3

class GlueCrawler:
    def __init__(self, crawler_name, crawler_s3_path, crawler_database_name, crawler_iam_role):
        self.crawler_name = crawler_name
        self.s3_path = crawler_s3_path
        self.database_name = crawler_database_name
        self.iam_role = crawler_iam_role

    def create_glue_crawler(self):
        glue_client = boto3.client('glue')

        # Check if the crawler already exists
        crawlers = glue_client.list_crawlers()['CrawlerNames']
        if self.crawler_name in crawlers:
            print(f"Crawler {self.crawler_name} already exists.")
        else:
            # Create the crawler
            response = glue_client.create_crawler(
                Name=self.crawler_name,
                Role=self.iam_role,
                DatabaseName=self.database_name,
                Description="Crawler for image data in Parquet format",
                Targets={'S3Targets': [{'Path': self.s3_path}]},
                TablePrefix='idp_',
                SchemaChangePolicy={
                    'UpdateBehavior': 'UPDATE_IN_DATABASE',
                    'DeleteBehavior': 'DEPRECATE_IN_DATABASE'
                }
            )
            print(f"Crawler {self.crawler_name} created successfully.")
    
    def start_crawler(self):
        glue_client = boto3.client('glue')
        try:
            glue_client.start_crawler(Name=self.crawler_name)
            print(f"Crawler {self.crawler_name} has been started.")
        except Exception as e:
            print(f"Failed to start crawler {self.crawler_name}. Error: {str(e)}")

    def run(self):
        self.create_glue_crawler()
        self.start_crawler()
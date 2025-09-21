import boto3

class GetXML():
    def __init__(self, extra_xmls):
        self.extra_xmls = extra_xmls
        self.s3 = boto3.client("s3")

    def compile_xml_list(self):
        xml_list = []
        for solver_xml in self.extra_xmls:
            if solver_xml['solver_xml'].startswith("s3://"):
                no_scheme = solver_xml['solver_xml'].replace("s3://", "", 1)
                bucket, key = no_scheme.split("/", 1)
                response = self.s3.get_object(Bucket=bucket, Key=key)
                xml_file = response["Body"].read().decode("utf-8")
                xml_list.append({
                    'xml_file': xml_file,
                    'output_xml_path': solver_xml['output_xml'],
                    'n5_path': solver_xml['n5_path']
                })
            else:  
                with open(solver_xml['solver_xml'], "r", encoding="utf-8") as f:
                    xml_file = f.read()
                    xml_list.append({
                        'xml_file': xml_file,
                        'output_xml_path': solver_xml['output_xml'],
                        'n5_path': solver_xml['n5_path']
                    })
        
        return xml_list

    def run(self):
        xml_list = self.compile_xml_list()

        return xml_list
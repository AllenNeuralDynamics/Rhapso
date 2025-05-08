import xml.etree.ElementTree as ET

class XMLParser:
    def __init__(self, xml_file):
        self.xml_file = xml_file

    def parse(self):
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        return root

    def parse_view_paths(self):
        root = self.parse()
        view_paths = {}
        for vip in root.findall(".//ViewInterestPointsFile"):
            setup_id = vip.attrib['setup']
            timepoint = int(vip.attrib['timepoint'])
            path = vip.text.strip()
            if path.endswith("/beads"):
                path = path[:-len("/beads")]
            key = (timepoint, setup_id)
            view_paths[key] = {'timepoint': timepoint, 'path': path, 'setup': setup_id}
        return view_paths

    def parse_timepoints(self):
        root = self.parse()
        timepoints = set()
        timepoints_list = root.find(".//Timepoints[@type='list']")
        if timepoints_list:
            for timepoint in timepoints_list.findall("Timepoint"):
                timepoints.add(int(timepoint.find("id").text))
        else:
            timepoint_range = root.find(".//Timepoints[@type='range']")
            if timepoint_range:
                first = int(timepoint_range.find("first").text)
                last = int(timepoint_range.find("last").text)
                timepoints = set(range(first, last + 1))
        return timepoints

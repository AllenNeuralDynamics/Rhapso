import Rhapso

# Call the xmlToDataframe function
myDataframe = Rhapso.xmlToDataframe("/mnt/c/Users/marti/Documents/Allen/repos/Rhapso-Sample-Data/IP_TIFF_XML/dataset.xml")
print('myDataframe = ', myDataframe)

# Call the runOverlapDetection function
output = Rhapso.runOverlapDetection(myDataframe)
print("results = ", output)
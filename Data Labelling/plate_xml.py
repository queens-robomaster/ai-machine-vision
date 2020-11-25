import xml.etree.ElementTree as ET
import os

NAME_OF_DIR = 'plate_visible_labelled'

# get the number of files in the directory
numFiles = int(len(os.listdir('./%s' % NAME_OF_DIR)))


# cycle through all files
for file in range(numFiles):
    # print('./plate_labelled/%0*d.xml' % (5, file))

    # get the tree of the XML file
    tree = ET.parse('./%s/%0*d.xml' % (NAME_OF_DIR, 5, file))
    root = tree.getroot()
    
    # loop through each name element found in the root
    for element in root.iter('name'):
        # replace the name with just plate
        element.text = 'plate'

    # save the xml file 
    tree.write('./%s/%0*d.xml' % (NAME_OF_DIR, 5, file))
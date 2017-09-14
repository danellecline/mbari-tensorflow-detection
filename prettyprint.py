 
import glob
import lxml.etree as etree
import tensorflow as tf

annotations_dir = '/Users/dcline/Dropbox/GitHub/mbari-tensorflow-detection/data/annotate/D0232_04HD/xmls' 
for path in glob.glob(annotations_dir + '/*.xml'):
    x = etree.parse(path)  
    pretty_xml_as_string = etree.tostring(x, pretty_print=True)
    with open(path, 'wb') as fid:
        fid.write(pretty_xml_as_string)
 

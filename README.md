# mbari-tensorflow-detection

RESTORE image from accidental delete !!!! using Windowss
 
 
mount the atlas deep learning share

run in pycharm on test directory


need to replace the image path with the correct one referenced to where the images will be for training ?

/raid/nyee/datasets/frcnn_data/D0232_04HD/xmls

Install Tensorflow from source
https://www.tensorflow.org/install/install_sources

Install your own dataset
https://github.com/tensorflow/models/blob/master/object_detection/g3doc/using_your_own_dataset.md

Replace globally 
</filename> with .png</filename>
and <folder>imgs</folder> with <folder>D0232_04HD</folder>

Checkout labelme output and see if that conforms to the PASCAL format expected in the 
create_pascal_tf_record script and don't force it to be the same if it's different.

instead, pass the parent directory of the xml file which is the basename for he tape number and 
use that. The dataset directory is the roof ot all the data

mkdir -p ~/Dropbox/GitHub/mbari-tensorflow-detection/data/annotate/D0232_04HD/imgs/
mkdir -p ~/Dropbox/GitHub/mbari-tensorflow-detection/data/annotate/D0232_04HD/xmls/
 
 
 
 We know that annotate output of labelme produces xmls with format
 
 <annotation>
  <filename>D0232_04HD_00-02-50.png</filename>
  <folder>D0232_04HD</folder>
  <source>
    <sourceImage>MBARI ROV Video</sourceImage>
    <sourceAnnotation>LabelMe Webtool</sourceAnnotation>
  </source>
  <object>
    <name>d</name>
    
 Nathan's output was
 <annotation verified="no">
  <folder>D0232_04HD</folder>
  <filename>D0232_04HD_00-03-50.png</filename>
  <path>/home/nathan/datasets/frcnn_data/D0232_04HD/imgs/D0232_04HD_00-03-50.png</path>
  <source>
    <database>Unknown</database>
  </source>
  <size>
    <width>960</width>
    <height>540</height>
    <depth>3</depth>
  </size>
  <segmented>0</segmented>
</annotation>

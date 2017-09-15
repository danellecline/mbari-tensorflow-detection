import sys
import os
import glob
import shutil
from bs4 import BeautifulSoup, Tag
import conf
import xml.dom.minidom

class LabelAggregator():
    """
    LabelAggregator searches for xml annotations and corresponding images and copies them 
    to a directory structure similar to VOC formatted data 
    """
    
    def __init__(self, annotation_dirs, output_dir, file_listing):
        """
        initialize the LabelAggregator
        :param annotation_dirs: directories to walk looking for xml annotation files 
        :param output_dir: output directory to store reorganized annotated data in 
        :param file_listing: file to dump the file listing to
        """
        self.annotation_dirs = [os.path.join(conf.BASE_DIR_RAW, i) for i in annotation_dirs] 
        self.output_dir = output_dir
        self.file_listing = file_listing

    def aggregate(self):
        """ recursively walk through directories searching for xml annotation files.
        In each file, replace the folder, path, etc. with the new data structure for export.
        Dump the file listings to a text file
         """
        
        # remove existing directory
        #if os.path.exists(self.output_dir):
        #    shutil.rmtree(self.output_dir)
            
        # setup new directory structure 
        annotate_dir = os.path.join(self.output_dir, conf.COLLECTION_NAME, conf.ANNOTATION_DIR)
        image_dir = os.path.join(self.output_dir, conf.COLLECTION_NAME, conf.PNG_DIR)
        self.check_dir(annotate_dir)
        self.check_dir(image_dir) 

        with open(os.path.join(self.output_dir, conf.COLLECTION_NAME, self.file_listing), 'w') as fid:

            for path in self.annotation_dirs:
                for xml_in in glob.iglob(path+ '/**/*.xml', recursive=True):
                    if self.file_standards(xml_in):

                        print('Processing {0}'.format(xml_in))

                        # find the source image in the xml_in file and replace
                        infile = open(xml_in, "r")
                        contents = infile.read() 
                        print('Parsing {0}'.format(xml_in))
                        soup = BeautifulSoup(contents, 'xml')

                        # rename the folder to the collection name
                        soup.folder.string = conf.COLLECTION_NAME

                        # remove the path
                        d = soup.find('path')
                        d.extract()

                        file_root = os.path.basename(xml_in)
                        file, ext = os.path.splitext(file_root)

                        # add the image tag to the source; here we assume it's the first 9 characters
                        image_tag = Tag(name="image")
                        image_tag.string = 'ROV Dive {0}'.format(file_root[0:9])
                        soup.source.insert(1, image_tag)

                        # append png to the filename and write out to the txt file
                        image_name =  file + '.png'
                        soup.filename.string = image_name
                        fid.write(file_root + '\n')
                        
                        xml_out = os.path.join(annotate_dir, file + ext)
                        f = open('tmp.xml', "w")
                        f.write(soup.decode_contents())
                        f.close()

                        # a bit of hacky workaround to print a better looking xml than what beautifulsoup produces
                        xmlf = xml.dom.minidom.parse('tmp.xml')  # or xml.dom.minidom.parseString(xml_string)
                        pretty_xml_as_string = xmlf.toprettyxml()
                        # remove empty lines
                        pretty_xml_as_string = os.linesep.join([s for s in pretty_xml_as_string.splitlines() if s.strip()])
                        with open(xml_out, 'w') as f2:
                            f2.write(pretty_xml_as_string)
                        f2.close()

                        # copy the source image to the correct spot
                        src = os.path.join(path, 'imgs', image_name)
                        dst = os.path.join(image_dir, image_name)
                        if not os.path.exists(dst):
                            shutil.copyfile(src, dst)
                        
                        soup.clear()

        fid.close()
 
 
    def check_dir(self, path):
        """
        Simply utility to check directory and if it doesn't exist make it and all its parent directories
        :param path: the path to check
        :return: 
        """
        if not os.path.exists(path):
            os.makedirs(path)

    def file_standards(self, filepath):
        """
        checks to make sure that the file should be processed. In this case:

        The file must be an xml file
        The filepath must not be from xml_preds
        One of the allowed directory names must be in the filepath
 
        :param filepath: the full filepath of the file  
        :returns: whether or not the file/filepath should be included 
        """
        file = os.path.basename(filepath)
        _, ext = os.path.splitext(file)
        is_xml = bool(ext == '.xml')
        not_pred = bool('xml_preds' not in filepath) 
        return (all([is_xml, not_pred]))

if __name__ == '__main__':
    
    train = LabelAggregator(conf.TRAIN_VID_KEYS, conf.BASE_DIR_CONVERT, 'train.txt')
    train.aggregate()

    train = LabelAggregator(conf.TEST_VID_KEYS, conf.BASE_DIR_CONVERT, 'test.txt')
    train.aggregate()

import sys
import os
import glob
import shutil
from bs4 import BeautifulSoup, Tag
import conf
import lxml.etree as etree

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
                for xml in glob.iglob(path+ '/**/*.xml', recursive=True): 
                    if self.file_standards(xml): 
                         
                        # find the source image in the xml file and replace  
                        infile = open(xml, "r")
                        contents = infile.read() 
                        print('Parsing {0}'.format(xml))
                        soup = BeautifulSoup(contents, 'xml') 
    
                        soup.path.string = conf.COLLECTION_NAME 
                        soup.path.folder = conf.PNG_DIR
                        
                        file_root = os.path.basename(xml)
                        file, ext = os.path.splitext(file_root)
    
                        image_name =  file + '.png'
                        soup.path.file = image_name
                        fid.write(file_root + '\n')
                        
                        xml_out = os.path.join(annotate_dir, file + ext)
                        print('Writing {0}'.format(xml_out))
                        f = open(xml_out, "w")
                        f.write(soup.decode_contents())
                        f.close() 
    
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

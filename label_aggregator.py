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
                        soup = BeautifulSoup(contents, 'xml')

                        # skip over annotations with no objects
                        if not soup.find('object'):
                            print('No objects found in {0} so skipping'.format(xml_in))
                            continue

                        # rename the folder to the collection name
                        soup.folder.string = conf.COLLECTION_NAME

                        # remove the path
                        d = soup.find('path')
                        d.extract()

                        # image size
                        image_height = int(soup.height.text)
                        image_width = int(soup.width.text)

                        scale_x = conf.TARGET_WIDTH/image_width
                        scale_y = conf.TARGET_HEIGHT/image_height

                        # rescale the annotations to the target image size
                        self.replace_tag(soup.xmax, scale_x)
                        self.replace_tag(soup.xmin, scale_x)
                        self.replace_tag(soup.ymax, scale_y)
                        self.replace_tag(soup.ymin, scale_y)

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
                        if os.path.exists(xml_out):
                            continue
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

                        #  rescale to the model size and copy the source image to the correct spot and
                        src = os.path.join(path, 'imgs', image_name)
                        dst = os.path.join(image_dir, image_name)

                        cmd = '/usr/local/bin/convert {0} -scale {1}x{2}\! "{3}"'.format(src,
                                                                                         conf.TARGET_WIDTH,
                                                                                         conf.TARGET_HEIGHT,
                                                                                         dst)
                        print('Executing {0}'.format(cmd))
                        os.system(cmd)
                        soup.clear()

        fid.close()

    def replace_tag(self, tag,  scale):
        num = int(int(tag.text) * scale)
        tag.string = '{0}'.format(num)

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

    test = LabelAggregator(conf.TEST_VID_KEYS, conf.BASE_DIR_CONVERT, 'test.txt')
    test.aggregate()

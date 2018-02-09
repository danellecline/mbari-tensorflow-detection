from collections import namedtuple

model_metadata = namedtuple("model_metadata", ["meta_arch", "feature_extractor", "proposals", "dir", "name", "resolution"])


class ModelMetadata():
  """
     ModelMetadata puts the model metadata as parsed from a string in to a class
     """
  meta_arch = 'Unknown'
  feature_arch = 'Unknown'
  proposal = 0
  image_resolution = 0
  name = 'None'

  def __init__(self, model_name):
    """
    initialize the ModelMetadata
    :param path: directories of model
    """
    self.name = model_name
    self.proposals = 0
    self.image_resolution = 0

    if 'resnet101' in model_name:
      self.feature_arch = 'Resnet 101'
    elif 'inception_v2' in model_name:
      self.feature_arch = 'Inception v2'
    if 'ssd' in model_name:
      self.meta_arch = 'SSD'
    elif 'faster_rcnn' in model_name:
      self.meta_arch = 'Faster RCNN'
    elif 'rfcn' in model_name:
      self.meta_arch = 'R-FCN'

    f = model_name.split('_')
    for j in f:
      if j.isnumeric():
        if self.meta_arch != 'SSD':
          self.proposals = int(j)
        else:
          self.image_resolution = int(j)

      # TODO add regex for resolution here

    print('Model architecture {0} feature extractor {1} proposals {2} image resolution {3}'.format(self.meta_arch, self.feature_arch, self.proposals, self.image_resolution))
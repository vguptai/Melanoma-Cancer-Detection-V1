import tensorflow as tf
import tarfile
from osUtils import *
import random
from six.moves import urllib
import sys
import numpy as np
from PIL import Image

def save_labels(FLAGS,image_map):
    print "Saving labels at:"+FLAGS.output_labels
    with tf.gfile.FastGFile(FLAGS.output_labels, 'w') as f:
      f.write('\n'.join(image_map.keys()) + '\n')


def get_bottleneck_path_new(image_path, label_name, bottleneck_dir):
	image_path = os.path.join(bottleneck_dir, label_name, image_path)
	return image_path + '.txt'


def get_image_path(image_lists, label_name, index, image_dir, category):
  """"Returns a path to an image for a label at the given index.
  Args:
    image_lists: Dictionary of training images for each label.
    label_name: Label string we want to get an image for.
    index: Int offset of the image we want. This will be moduloed by the
    available number of images for the label, so it can be arbitrarily large.
    image_dir: Root folder string of the subfolders containing the training
    images.
    category: Name string of set to pull images from - training, testing, or
    validation.
  Returns:
    File system path string to an image that meets the requested parameters.
  """
  if label_name not in image_lists:
    tf.logging.fatal('Label does not exist %s.', label_name)
  label_lists = image_lists[label_name]
  if category not in label_lists:
    tf.logging.fatal('Category does not exist %s.', category)
  category_list = label_lists[category]
  if not category_list:
    tf.logging.fatal('Label %s has no images in the category %s.',
                     label_name, category)
  mod_index = index % len(category_list)
  base_name = category_list[mod_index]
  sub_dir = label_lists['dir']
  full_path = os.path.join(image_dir, sub_dir, base_name)
  return full_path

def create_bottleneck_file_new(bottleneck_path, image_data, sess, inceptionV3Model):
  """Create a single bottleneck file."""
  print('Creating bottleneck at ' + bottleneck_path)
  try:
    bottleneck_values = inceptionV3Model.run_bottleneck_on_image(
        sess, image_data)
  except Exception as e:
	print e
	raise RuntimeError('Error during processing file %s' % bottleneck_path)

  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


"""
Retrieves or calculates bottleneck values for an image.
Args:
    sess: The current active TensorFlow Session.
    image_data:image data.
    image_index: location of image in the list
    image_category: train/test
    bottleneck_dir: Folder string holding cached files of bottleneck values.
Returns:
    Numpy array of values produced bget_bottleneck_pathy the bottleneck layer for the image.
"""
def get_or_create_bottleneck_new(sess, image_index, image_data, image_category, bottleneck_dir, inceptionV3Model):
  sub_dir_path = os.path.join(bottleneck_dir, image_category)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path_new(image_index, image_category, bottleneck_dir)

  if not os.path.exists(bottleneck_path):
    create_bottleneck_file_new(bottleneck_path, image_data, sess, inceptionV3Model)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    print('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file_new(bottleneck_path, image_data, sess, inceptionV3Model)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values

"""
Retrieves bottleneck values for images.
Args:
    sess: Current TensorFlow Session.
    image_data: array of image data.
    batch_offset: offset of image batch in the list
    image_category: train/test
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    Returns:
    List of bottleneck arrays.
"""
def get_random_cached_bottlenecks(sess, batch_offset, image_data,image_category,bottleneck_dir, inceptionV3Model):
  bottlenecks = []
  for i in range(image_data.shape[0]):
      #print str(i)+"/"+str(image_data.shape[0])
      bottleneck = get_or_create_bottleneck_new(sess,str(batch_offset+i),image_data[i],image_category,bottleneck_dir, inceptionV3Model)
      bottlenecks.append(bottleneck)
  return bottlenecks

def get_random_distorted_bottlenecks(
    sess, image_paths, inceptionV3Model):
  """Retrieves bottleneck values for training images, after distortions.
  If we're training with distortions like crops, scales, or flips, we have to
  recalculate the full model for every image, and so we can't use cached
  bottleneck values. Instead we find random images for the requested category,
  run them through the distortion graph, and then the full graph to get the
  bottleneck results for each.
  Args:
    sess: Current TensorFlow Session.
    image_paths: List of image paths.
    input_jpeg_tensor: The input layer we feed the image data to.
    distorted_image: The output node of the distortion graph.
  Returns:
    List of bottleneck arrays and their corresponding ground truths.
  """
  bottlenecks = []
  for i in range(len(image_paths)):
      image = Image.open(image_paths[i])
      image_data = image.convert('RGB')
      distorted_image_data = inceptionV3Model.distort_image(sess,image_data)
      try:
        bottleneck = inceptionV3Model.run_bottleneck_on_image(sess, distorted_image_data)
        bottlenecks.append(bottleneck)
      except Exception as e:
        print e
        raise RuntimeError('Error during processing file %s' % image_paths[i])
  return bottlenecks

def should_distort_images(FLAGS):
    """Whether any distortions are enabled, from the input flags.
    Args:
    flip_left_right: Boolean whether to randomly mirror images horizontally.
    random_crop: Integer percentage setting the total margin used around the
    crop box.
    random_scale: Integer percentage of how much to vary the scale by.
    random_brightness: Integer range to randomly multiply the pixel values by.
    Returns:
    Boolean value indicating whether any distortions should be applied.
    """
    if(FLAGS.apply_distortions):
        return (FLAGS.flip_left_right or (FLAGS.random_crop != 0) or (FLAGS.random_scale != 0) or (FLAGS.random_brightness != 0))
    else:
        return False

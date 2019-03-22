"""
"""
import os
import numpy as np
import tensorflow as tf

HEIGHT = 64
WIDTH = 64
DEPTH = 3
NUM_JOINTS = 36
HM_SIZE = 64
PI = 3.1412

class DataSet(object):
  """Generate data set """

  def __init__(self, data_dir, subset='train', use_distortion=True, seq_length=4):
    self.data_dir = data_dir
    self.subset = subset
    self.use_distortion = use_distortion
    self.writer = tf.summary.FileWriter(data_dir)
    self.seq_length = seq_length

  def get_filenames(self):
    if self.subset in ['train', 'test', 'eval']:
      return [os.path.join(self.data_dir, self.subset + '.tfrecords')]
    else:
      raise ValueError('Invalid data subset "%s"' % self.subset)

  def _makeGaussian(self, HEIGHT, WIDTH, sigma = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    sigma is full-WIDTH-half-maximum, which
    can be thought of as an effective radius.
    """
    if not np.array_equal(center,[-1,-1]):
      x = np.arange(0, WIDTH, 1, float)
      y = np.arange(0, HEIGHT, 1, float)[:, np.newaxis]
      if center is None:
        x0 =  WIDTH // 2
        y0 = HEIGHT // 2
      else:
        x0 = center[0]
        y0 = center[1]
      return tf.exp(tf.cast(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2,tf.float32))
    else:
      return np.zeros([WIDTH, HEIGHT])

  def generate_hm(self, center, std, size):
    """ Make 2D gaussian kernal of heatmaps """
    distribution = tf.distributions.Normal(loc=tf.cast(center,tf.float32), scale=[std,std])
    vals = tf.range(size, dtype=tf.float32)
    x,y = tf.meshgrid(vals,vals)
    x = tf.reshape(x,[-1])
    y = tf.reshape(y,[-1])
    elem = tf.stack([x,y],axis=1)
    hm = tf.reshape(tf.reduce_prod(tf.map_fn(fn=lambda x:distribution.prob([x[0],x[1]]),elems=elem),axis=1),[size,size])
    # gauss_kernel = tf.einsum('i,j->ij', values,values)
    return hm

  def parser(self, serialized_example):
    """Parses a single tf.Example into image and label tensors."""

    feature = {}
    for j in range(self.seq_length):
      feature['image_' + str(j)] = tf.FixedLenFeature([], tf.string)
      feature['joints_' + str(j)] = tf.FixedLenFeature([NUM_JOINTS*2], tf.int64)
      #feature['occlusion_' + str(j)] = tf.FixedLenFeature([NUM_JOINTS], tf.int64)
      #feature['depth_' + str(j)] = tf.FixedLenFeature([], tf.string)

    features = tf.parse_single_example(
        serialized_example,
        features=feature)
    images = []
    hms = []
    weights = []
    depths = []

    delta_brightness = tf.random_uniform([1], -0.3, 0.3)
    contrast_factor = tf.random_uniform([1], 0.2, 1.8)
    rotation_angle = tf.random_uniform([1],-PI*0.1,PI*0.1)

    for j in range(self.seq_length):
      hm = []
      image = tf.decode_raw(features['image_' + str(j)], tf.uint8)
      image.set_shape([DEPTH * HEIGHT * WIDTH])
      image = tf.cast(tf.reshape(image, [HEIGHT, WIDTH, DEPTH]),tf.float32)/255.0

      label = tf.cast(features['joints_' + str(j)], tf.int32)
      joints = tf.reshape(label, [NUM_JOINTS, 2])

      #weight = tf.cast(features['occlusion_' + str(j)], tf.float32)
      #weight = tf.reshape(weight, [NUM_JOINTS])
      
      #depth = tf.decode_raw(features['depth_' + str(j)], tf.float32)
      #depth.set_shape([HEIGHT * WIDTH])
      #depth = tf.cast(tf.reshape(depth, [HEIGHT, WIDTH]),tf.float32)

      # hm_list = np.zeros((HEIGHT, WIDTH, NUM_JOINTS), dtype = np.float32)
      hm_list = []
      for j in range(NUM_JOINTS):
        # hm_list.append(self.generate_hm(center=joints[j], std=1.0, size=HM_SIZE))
        s = int(np.sqrt(HM_SIZE) * HM_SIZE * 10 / 4096) + 2
        heatmap = self._makeGaussian(HM_SIZE, HM_SIZE, sigma= s, center=joints[j])
        # if self.subset == 'train' and self.use_distortion:
          # heatmap = tf.contrib.image.rotate(heatmap, rotation_angle)
        hm_list.append(heatmap)

      hm = tf.stack(hm_list)
      
      # Custom preprocessing.
      image = self.preprocess(image, delta_brightness, contrast_factor[0], rotation_angle)

      images.append(image)
      hms.append(hm)
      #weights.append(weight)
      #depths.append(depth)

    return tf.stack(images), tf.stack(hms), tf.stack(weights), tf.stack(depths)

  def img_to_tensorboard(self, img, hm):
    """ Returns operations to visualize image and heatmaps """

    image_summary_op = tf.summary.image('images', tf.reshape(img[0],[-1, HEIGHT, WIDTH, DEPTH]))
    hm_summary_op = tf.summary.image('heatmaps', tf.reshape(tf.reduce_sum(hm[0], 0), [-1, HM_SIZE, HM_SIZE, 1]))
    self.summary_write(image_summary_op)
    self.summary_write(hm_summary_op)
    return image_summary_op, hm_summary_op

  def summary_write(self, operation):
    with tf.Session() as sess:
      summary = sess.run(operation)
      self.writer.add_summary(summary)

  def make_batch(self, batch_size):
    """Read the images and labels from 'filenames'."""
    filenames = self.get_filenames()
    # Repeat infinitely.
    dataset = tf.data.TFRecordDataset(filenames).repeat()

    # Parse records.
    dataset = dataset.map(
        self.parser, num_parallel_calls=batch_size)

    # Potentially shuffle records.
    if self.subset == 'train':
      min_queue_examples = int(
          DataSet.num_examples_per_epoch(self.subset) * 0.4)
      # Ensure that the capacity is sufficiently large to provide good random
      # shuffling.
      dataset = dataset.shuffle(buffer_size=min_queue_examples + 3 * batch_size)

    # Batch it up.
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size= 3 * batch_size)
    iterator = dataset.make_one_shot_iterator()
    image_batch, label_batch, occlusion_batch, depth_batch = iterator.get_next()

    return image_batch, label_batch, occlusion_batch, depth_batch

  def preprocess(self, image, delta_brightness, contrast_factor, rotation_angle):
    """Preprocess a single image in [HEIGHT, WIDTH, depth] layout."""
    if self.subset == 'train' and self.use_distortion:
      image = tf.image.adjust_contrast(image,contrast_factor)
      image = tf.clip_by_value(image, 0.0, 1.0)
      image = tf.image.adjust_brightness(image,delta_brightness)
      image = tf.clip_by_value(image, 0.0, 1.0)
      # image = tf.contrib.image.rotate(image, rotation_angle)
    return image

  @staticmethod
  def num_examples_per_epoch(subset='train'):
    if subset == 'train':
      return 1
    elif subset == 'test':
      return 1
    elif subset == 'eval':
      return 16
    else:
      raise ValueError('Invalid data subset "%s"' % subset)

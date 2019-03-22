"""
*   HourGlass TensorFlow Implementation
*
*   Copyright (C) 2018  Ayush Gaud
*
*   This program is free software: you can redistribute it and/or modify
*   it under the terms of the GNU General Public License as published by
*   the Free Software Foundation, either version 3 of the License, or
*   (at your option) any later version.
*
*   This program is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*   GNU General Public License for more details.
*
*   You should have received a copy of the GNU General Public License
*   along with this program.  If not, see <http://www.gnu.org/licenses/>.
*
*   Author Ayush Gaud <ayush.gaud[at]gmail.com>

"""

import argparse
import functools
import itertools
import os

import data_parser
import hg_model
import utils
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import inspect
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)


def get_model_fn(num_gpus, variable_strategy, num_workers):
  """Returns a function that will build the hourglass model."""

  def _hg_model_fn(features, labels, mode, params):
    """ HG model body.

    Support single host, one or more GPU training. Parameter distribution can
    be either one of the following scheme.
    1. CPU is the parameter server and manages gradient updates.
    2. Parameters are distributed evenly across all GPUs, and the first GPU
       manages gradient updates.

    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    weight_decay = params.weight_decay
    momentum = params.momentum
    decay_factor = params.decay_factor
    decay_step = params.decay_step
    init_learning_rate = params.init_learning_rate
    num_stacks = params.num_stacks
    num_joints = params.num_joints

    tower_features = features
    if mode == tf.estimator.ModeKeys.PREDICT:
      if num_gpus < 1:
        tower_labels = [None]
      else:
        tower_labels = [None for i in range(num_gpus)]
    else:
      tower_labels = labels

    tower_losses = []
    tower_gradvars = []
    tower_preds = []

    # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
    # on CPU. The exception is Intel MKL on CPU which is optimal with
    # channels_last.
    data_format = params.data_format
    if not data_format:
      if num_gpus == 0:
        data_format = 'channels_last'
      else:
        data_format = 'channels_first'

    if num_gpus == 0:
      num_devices = 1
      device_type = 'cpu'
    else:
      num_devices = num_gpus
      device_type = 'gpu'

    for i in range(num_devices):
      worker_device = '/{}:{}'.format(device_type, i)
      if variable_strategy == 'CPU':
        device_setter = utils.local_device_setter(
            worker_device=worker_device)
      elif variable_strategy == 'GPU':
        device_setter = utils.local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
      if mode == tf.estimator.ModeKeys.TRAIN:
        batch_size = params.train_batch_size / num_devices
      else:
        batch_size = params.eval_batch_size / num_devices

      with tf.variable_scope('hg', reuse=bool(i != 0)):
        with tf.name_scope('tower_%d' % i) as name_scope:
          with tf.device(device_setter):
            loss, gradvars, preds = _tower_fn(
                mode, weight_decay, tower_features[i][0], tower_labels[i],
                data_format, params.batch_norm_decay,
                params.batch_norm_epsilon, params.num_stacks, params.num_out, params.n_low, params.num_joints, batch_size,params.seq_length)
            tower_losses.append(loss)
            tower_gradvars.append(gradvars)
            tower_preds.append(preds)
            if i == 0:
              # Only trigger batch_norm moving mean and variance update from
              # the 1st tower. Ideally, we should grab the updates from all
              # towers but these stats accumulate extremely fast so we can
              # ignore the other stats from the other towers without
              # significant detriment.
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                             name_scope)

    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:

      # Now compute global loss and gradients.
      gradvars = []
      with tf.name_scope('gradient_averaging'):
        all_grads = {}
        for grad, var in itertools.chain(*tower_gradvars):
          if grad is not None:
            all_grads.setdefault(var, []).append(grad)
        for var, grads in six.iteritems(all_grads):
          # Average gradients on the same device as the variables
          # to which they apply.
          with tf.device(var.device):
            if len(grads) == 1:
              avg_grad = grads[0]
            else:
              avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
          gradvars.append((avg_grad, var))

      # Device that runs the ops to apply global gradient updates.
      consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
      with tf.device(consolidation_device):

        learning_rate = tf.train.exponential_decay(init_learning_rate, tf.train.get_global_step(), decay_step, decay_factor, staircase=True, name= 'learning_rate')

        loss = tf.reduce_mean(tower_losses, name='loss')

        examples_sec_hook = utils.ExamplesPerSecondHook(
            params.train_batch_size, every_n_steps=10)

        tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}

        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=100)

        train_hooks = [logging_hook, examples_sec_hook]

        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

        if params.sync:
          optimizer = tf.train.SyncReplicasOptimizer(
              optimizer, replicas_to_aggregate=num_workers)
          sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
          train_hooks.append(sync_replicas_hook)

        # Create single grouped train op
        train_op = [
            optimizer.apply_gradients(
                gradvars, global_step=tf.train.get_global_step())
        ]
        
        train_op.extend(update_ops)
        train_op = tf.group(*train_op)

        predictions = {
            'heatmaps':
                tf.concat([p['heatmaps'] for p in tower_preds], axis=0),
            'images':
                tf.concat([i for i in tower_features], axis=0)
        }
        if mode==tf.estimator.ModeKeys.EVAL:
          hm = predictions['heatmaps']
          stacked_labels = tf.concat(labels[0][0][0], axis=0)
          
          gt_labels = tf.transpose(stacked_labels,[1,0,3,4,2])

          joint_accur = []
          for j in range(params.seq_length):
            for i in range(params.num_joints):
              joint_accur.append(_pck_hm(hm[j,:,-1, :, :,i], gt_labels[j,:, :, :, i], params.eval_batch_size/num_devices))
          accuracy = tf.stack(joint_accur)
          metrics = {'Mean Pixel Error': tf.metrics.mean(accuracy)}
          tf.logging.info('Accuracy op computed')
        else:
          metrics = None
    
    else:
      train_op = None
      loss = None
      train_hooks = None
      metrics = None
      predictions = {
          'heatmaps':
              tf.concat([p['heatmaps'] for p in tower_preds], axis=0),
          'images':
              tf.concat([i for i in tower_features], axis=0)
      }
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=train_hooks,
        eval_metric_ops=metrics)

  return _hg_model_fn

def _pck_hm(hm, gthm, batch_size, alpha=0.031):
  
  pck = tf.to_float(0)
  num_valid_joints = tf.to_float(0)
  for i in range(int(batch_size)):
      l2_norm = tf.norm(tf.stack(argmax_2d(hm[i,:,:]) - argmax_2d(gthm[i,:,:])))
      l2_norm = tf.cond(tf.equal(tf.reduce_sum(gthm[i,:,:]), 0.0), lambda: 0.0, lambda: l2_norm)
      is_valid = tf.cond(tf.greater(64.0*alpha, l2_norm),lambda: 1.0, lambda: 0.0)
      pck = tf.add(pck, is_valid)
      num_valid_joints = tf.cond(tf.equal(tf.reduce_sum(gthm[i,:,:]), 0.0), lambda: num_valid_joints, lambda: tf.add(1.0,num_valid_joints))
  return tf.divide(pck,num_valid_joints)

def argmax_2d(matrix):

  vector = tf.reshape(matrix,[-1])
  index = tf.argmax(vector)
  shape = tf.constant([matrix.get_shape().as_list()[0], matrix.get_shape().as_list()[1]],dtype=tf.int64)
  return tf.cast(tf.unravel_index(index, shape),tf.float32)

def _tower_fn(mode, weight_decay, feature, label, data_format,
              batch_norm_decay, batch_norm_epsilon, num_stacks, num_out, n_low, num_joints, batch_size, seq_length=1):
  """Build computation tower.

  Args:
    is_training: true if is training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    data_format: channels_last (NHWC) or channels_first (NCHW).
    num_layers: number of layers, an int.
    batch_norm_decay: decay for batch normalization, a float.
    batch_norm_epsilon: epsilon for batch normalization, a float.
    batch_size

  Returns:
    A tuple with the loss for the tower, the gradients and parameters, and
    predictions.

  """
  is_training = (mode == tf.estimator.ModeKeys.TRAIN)
  model = hg_model.HourGlass(
      batch_norm_decay=batch_norm_decay,
      batch_norm_epsilon=batch_norm_epsilon,
      is_training=is_training,
      data_format=data_format,
      num_stacks=num_stacks, 
      num_out=num_out, 
      n_low=n_low,
      num_joints=num_joints,
      batch_size=batch_size)

  cells = []
  for i in range(n_low):
    cells.append(tuple([model._recurrent_cell(i+1, num_out,'cell_' + str(i) + str(j)) for j in xrange(num_stacks)]))
  input_states = [[None]*num_stacks for _ in range(n_low)]
  logits = []
  for j in range(seq_length):
    with tf.variable_scope('seq', reuse=bool(j != 0)):
      output, output_states = model.model(inputs=feature[:,j],cells=cells,input_states=input_states)
      logits.append(output)
      input_states = output_states
  logits = tf.stack(logits)
  tower_pred = {
      'heatmaps': tf.nn.sigmoid(logits)
  }
  
  if mode != tf.estimator.ModeKeys.PREDICT:
    tower_loss = tf.nn.sigmoid_cross_entropy_with_logits(
         logits=logits, labels=tf.transpose(tf.stack ([label[0][0][0] for i in range(num_stacks)]),[2,1,0,4,5,3]))

    tower_loss = tf.reduce_mean(tower_loss)
    #tower_loss = tf.losses.mean_squared_error(tf.transpose(tf.stack ([label[0][0][0] for i in range(num_stacks)]),[2,1,0,4,5,3]),logits)

    model_params = tf.trainable_variables()
    #tower_loss += weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in model_params])

    tower_grad = tf.gradients(tower_loss, model_params)
  else:
    tower_loss = [None]
    tower_grad = [None]
    model_params = tf.trainable_variables()

  return tower_loss, zip(tower_grad, model_params), tower_pred


def input_fn(data_dir,
             subset,
             num_shards,
             batch_size,
             seq_length=4,
             use_distortion_for_training=True):
  """Create input graph for model.

  Args:
    data_dir: Directory where TFRecords representing the dataset are located.
    subset: one of 'train', 'validate' and 'eval'.
    num_shards: num of towers participating in data-parallel training.
    batch_size: total batch size for training to be divided by the number of
    shards.
    use_distortion_for_training: True to use distortions.
  Returns:
    two lists of tensors for features and labels, each of num_shards length.
  """
  with tf.device('/gpu:0' if num_shards >= 1 else '/cpu:0'):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = data_parser.DataSet(data_dir, subset, use_distortion, seq_length)
    image_batch, label_batch, occlusion_batch, depth_batch = dataset.make_batch(batch_size)

    # Note that passing num=batch_size is safe here, even though
    # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
    # examples. This is because it does so only when repeating for a limited
    # number of epochs, but our dataset repeats forever.
    image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
    label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
    occlusion_batch = tf.unstack(occlusion_batch, num=batch_size, axis=0)
    depth_batch = tf.unstack(depth_batch, num=batch_size, axis=0)
    feature_shards = [[] for i in range(num_shards)]
    label_shards = [[] for i in range(num_shards)]
    occlusion_shards = [[] for i in range(num_shards)]
    depth_shards = [[] for i in range(num_shards)]
    skip = batch_size/num_shards
    for idx in range(num_shards):
      feature_shards[idx].append(tf.parallel_stack(image_batch[idx*skip:(idx+1)*skip]))
      label_shards[idx].append([[tf.parallel_stack(label_batch[idx*skip:(idx+1)*skip])], [tf.parallel_stack(occlusion_batch[idx*skip:(idx+1)*skip])], [tf.parallel_stack(depth_batch[idx*skip:(idx+1)*skip])]])

    return feature_shards, label_shards


def get_estimator_fn(num_gpus,
                      variable_strategy,
                      run_config,
                      hparams):
  """Returns an Experiment function.
  
  """
  estimator = tf.estimator.Estimator(
      model_fn=get_model_fn(num_gpus, variable_strategy,
                            run_config.num_worker_replicas or 1),
      config=run_config,
      params=hparams)

  return estimator

def get_train_spec(num_gpus, hparams, data_dir, use_distortion_for_training):
  
  train_input_fn = functools.partial(
      input_fn,
      data_dir,
      subset='train',
      num_shards=num_gpus,
      batch_size=hparams.train_batch_size,
      seq_length=hparams.seq_length,
      use_distortion_for_training=use_distortion_for_training)
  
  train_steps = hparams.train_steps

  train_spec = tf.estimator.TrainSpec(
    input_fn=train_input_fn,
    max_steps=train_steps)

  return train_spec

def get_eval_spec(num_gpus, hparams, data_dir):

  eval_input_fn = functools.partial(
      input_fn,
      data_dir,
      subset='eval',
      batch_size=hparams.eval_batch_size,
      seq_length=hparams.seq_length,
      num_shards=num_gpus)
  
  num_eval_examples = data_parser.DataSet.num_examples_per_epoch('eval')
  if num_eval_examples % hparams.eval_batch_size != 0:
    raise ValueError(
        'validation set size must be multiple of eval_batch_size')

  eval_steps = num_eval_examples // hparams.eval_batch_size

  eval_spec = tf.estimator.EvalSpec(
    input_fn=eval_input_fn,
    steps=eval_steps)

  return eval_spec

def draw_predictions(lbls,output_hm,output_img,seq_length,wireframe=True):
  ''' Draw predictions, ground truth heatmap and wireframe '''
  for j in range(seq_length):
    plt.rcParams['figure.figsize'] = [10, 10]
    plt.subplot(1,3,1)
    plt.imshow(np.sum(output_hm[j],axis=2))
    plt.subplot(1,3,2)
    plt.imshow(np.sum(lbls[0][0][j],axis=0))
    plt.subplot(1,3,3)
    pred_list = []
    gt_list = []
    for i in range(36):
        gt_joint = np.unravel_index(lbls[0][0][j][i].argmax(),[64,64])
        joint = np.unravel_index(output_hm[j,:,:,i].argmax(),[64,64])
        plt.scatter(x=joint[1], y=joint[0], c='r', s=20)
        plt.scatter(x=gt_joint[1], y=gt_joint[0], c='b', s=20)
        gt_list.append(gt_joint)
        pred_list.append(joint)
    
    # Plot wireframe
    plt_joint = np.array(pred_list)
    if(wireframe):
      for i in range(8):
          plt.plot([plt_joint[i,1], plt_joint[i+18,1]],[plt_joint[i,0], plt_joint[i+18,0]],color='blue', linewidth=2)
      for i in range(15):
          plt.plot([plt_joint[i,1], plt_joint[i+1,1]],[plt_joint[i,0], plt_joint[i+1,0]],color='red', linewidth=2)
      plt.plot([plt_joint[0,1], plt_joint[15,1]],[plt_joint[0,0], plt_joint[15,0]],color='red', linewidth=2)
      for i in range(18,33):
          plt.plot([plt_joint[i,1], plt_joint[i+1,1]],[plt_joint[i,0], plt_joint[i+1,0]],color='green', linewidth=2)
      plt.plot([plt_joint[18,1], plt_joint[33,1]],[plt_joint[18,0], plt_joint[33,0]],color='green', linewidth=2)
    plt.imshow(output_img[j].astype(np.uint8))
    plt.show()

def main(job_dir, data_dir, num_gpus, variable_strategy,
         use_distortion_for_training, log_device_placement, num_intra_threads,
         **hparams):
  # The env variable is on deprecation path, default is set to off.
  # os.environ['TF_SYNC_ON_FINISH'] = '0'
  # os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

  # Session configuration.
  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      log_device_placement=log_device_placement,
      intra_op_parallelism_threads=num_intra_threads,
      gpu_options=tf.GPUOptions(force_gpu_compatible=True,allow_growth=True))
  config = tf.estimator.RunConfig(
      session_config=sess_config, model_dir=job_dir, save_summary_steps=10)

  hparams_ = tf.contrib.training.HParams(
            is_chief=config.is_chief,
            **hparams)
  if(hparams_.mode != 'test'):
    tf.estimator.train_and_evaluate(
        estimator=get_estimator_fn(num_gpus=num_gpus, variable_strategy=variable_strategy,
                          run_config=config,
                          hparams=hparams_),
        train_spec=get_train_spec(num_gpus, hparams_, data_dir, use_distortion_for_training),
        eval_spec=get_eval_spec(num_gpus,hparams_, data_dir)
        )
  else:
    print("Creating estimator object for testing")

    test_input_fn = functools.partial(
                  input_fn,
                  data_dir,
                  subset='test',
                  batch_size=hparams_.eval_batch_size,
                  seq_length=hparams_.seq_length,
                  num_shards=num_gpus)

    predict = get_model_fn(num_gpus, variable_strategy, 0)
    features, labels = test_input_fn()
    # Rebuild the model
    predictions = predict(features, labels, tf.estimator.ModeKeys.PREDICT, hparams_).predictions
    # Manually load the latest checkpoint
    saver = tf.train.Saver()
    with tf.Session(config=sess_config) as sess:
      ckpt = tf.train.get_checkpoint_state(job_dir)
      saver.restore(sess, ckpt.model_checkpoint_path)
      acc_tot = []
      hm = predictions['heatmaps']
      stacked_labels = tf.concat(labels[0][0][0], axis=0)
      gt_labels = tf.transpose(stacked_labels,[1,0,3,4,2])
      joint_accur = []
      j = hparams_.seq_length - 1
      #for j in range(hparams_.seq_length):
      for i in range(hparams_.num_joints):
        joint_accur.append(_pck_hm(hm[j,:,-1, :, :,i], gt_labels[j,:, :, :, i], hparams_.eval_batch_size,alpha=0.031))
      accuracy = tf.reduce_mean(tf.stack(joint_accur))
      for _ in range(hparams_.eval_steps):
        try:
            accu, pred_dict, lbls = sess.run([accuracy, predictions, labels[0][0][0]])
            acc_tot.append(accu)
            print(accu)
            if hparams_.draw_predictions:
              output_hm = pred_dict['heatmaps'][:,0,-1]
              output_img = pred_dict['images'][0,0]*255
              draw_predictions(lbls,output_hm,output_img,hparams_.seq_length)
        except tf.errors.OutOfRangeError:
            break
      print("Final PCK %.2f" % (np.mean(acc_tot)*100.0))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--data-dir',
      type=str,
      required=True,
      help='The directory where the  input data is stored.')
  parser.add_argument(
      '--job-dir',
      type=str,
      required=True,
      help='The directory where the model will be stored.')
  parser.add_argument(
      '--variable-strategy',
      choices=['CPU', 'GPU'],
      type=str,
      default='CPU',
      help='Where to locate variable operations')
  parser.add_argument(
      '--num-gpus',
      type=int,
      default=1,
      help='The number of gpus used. Uses only CPU if set to 0.')
  parser.add_argument(
      '--num-layers',
      type=int,
      default=44,
      help='The number of layers of the model.')
  parser.add_argument(
      '--train-steps',
      type=int,
      default=80000,
      help='The number of steps to use for training.')
  parser.add_argument(
      '--train-batch-size',
      type=int,
      default=1,
      help='Batch size for training.')
  parser.add_argument(
      '--eval-batch-size',
      type=int,
      default=1,
      help='Batch size for validation.')
  parser.add_argument(
      '--momentum',
      type=float,
      default=0.9,
      help='Momentum for MomentumOptimizer.')
  parser.add_argument(
      '--weight-decay',
      type=float,
      default=2e-4,
      help='Weight decay for convolutions.')
  parser.add_argument(
      '--init-learning-rate',
      type=float,
      default=2.5e-4,
      help="""\
      This is the inital learning rate value. The learning rate will decrease
      during training. For more details check the model_fn implementation in
      this file.\
      """)
  parser.add_argument(
      '--decay-factor',
      type=float,
      default=0.96,
      help='decay factor of learning rate.')
  parser.add_argument(
      '--decay-step',
      type=float,
      default=1e4,
      help='Learning rate deacy step.')
  parser.add_argument(
      '--use-distortion-for-training',
      type=bool,
      default=True,
      help='If doing image distortion for training.')
  parser.add_argument(
      '--sync',
      action='store_true',
      default=False,
      help="""\
      If present when running in a distributed environment will run on sync mode.\
      """)
  parser.add_argument(
      '--num-intra-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for intra-op parallelism. When training on CPU
      set to 0 to have the system pick the appropriate number or alternatively
      set it to the number of physical CPU cores.\
      """)
  parser.add_argument(
      '--num-inter-threads',
      type=int,
      default=0,
      help="""\
      Number of threads to use for inter-op parallelism. If set to 0, the
      system will pick an appropriate number.\
      """)
  parser.add_argument(
      '--data-format',
      type=str,
      default=None,
      help="""\
      If not set, the data format best for the training device is used. 
      Allowed values: channels_first (NCHW) channels_last (NHWC).\
      """)
  parser.add_argument(
      '--log-device-placement',
      action='store_true',
      default=False,
      help='Whether to log device placement.')
  parser.add_argument(
      '--batch-norm-decay',
      type=float,
      default=0.997,
      help='Decay for batch norm.')
  parser.add_argument(
      '--batch-norm-epsilon',
      type=float,
      default=1e-5,
      help='Epsilon for batch norm.')
  parser.add_argument(
      '--num-stacks',
      type=int,
      default=2,
      help='Number of hourglass stacks.')
  parser.add_argument(
      '--num-out',
      type=int,
      default=256,
      help='Feature size.')
  parser.add_argument(
      '--n-low',
      type=int,
      default=4,
      help='Number of lower branches.')
  parser.add_argument(
      '--num-joints',
      type=int,
      default=36,
      help='Number of joints.')
  parser.add_argument(
      '--mode',
      type=str,
      default='train',
      help='Mode of operation e.g. train/test.')
  parser.add_argument(
      '--seq-length',
      type=int,
      default=4,
      help='Sequence length.')
  parser.add_argument(
      '--draw-predictions',
      type=bool,
      default=False,
      help='Draw Predictions')
  parser.add_argument(
      '--eval-steps',
      type=int,
      default=1,
      help='Number of evaluation steps')
  args = parser.parse_args()

  if args.num_gpus > 0:
    assert tf.test.is_gpu_available(), "Requested GPUs but none found."
  if args.num_gpus < 0:
    raise ValueError(
        'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
  if args.num_gpus == 0 and args.variable_strategy == 'GPU':
    raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                     '--variable-strategy=CPU.')
  if (args.num_layers - 2) % 6 != 0:
    raise ValueError('Invalid --num-layers parameter.')
  if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
    raise ValueError('--train-batch-size must be multiple of --num-gpus.')
  if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
    raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

  main(**vars(args))

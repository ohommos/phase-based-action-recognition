import tensorflow as tf
import numpy as np
from scipy.stats import unitary_group as ug
import cv2
import hyperparameters as hp
from hyperparameters import tbprint
from complexbn import ComplexBatchNormalization
from complexinit import IndependentFilters
from complexbn import ComplexBN
import keras.backend as K
import keras


""" Helper functions"""
def put_kernels_on_grid (kernel, pad=1):

  '''Visualize conv. filters as an image (mostly for the 1st layer).
  Arranges filters into a grid, with some paddings between adjacent filters.
  Args:
    kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
    pad:               number of black pixels around each filter (between them)
  Return:
    Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
  '''
  # get shape of the grid. NumKernels == grid_Y * grid_X
  def factorization(n):
    for i in range(int(np.sqrt(float(n))), 0, -1):
      if n % i == 0:
        if i == 1: print('Who would enter a prime number of filters')
        return (i, int(n / i))
  (grid_Y, grid_X) = factorization (kernel.get_shape()[3].value)
  # print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3].value, grid_Y, grid_X))

  x_min = tf.reduce_min(kernel)
  x_max = tf.reduce_max(kernel)
  kernel = (kernel - x_min) / (x_max - x_min)

  # pad X and Y
  x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

  # X and Y dimensions, w.r.t. padding
  Y = kernel.get_shape()[0] + 2 * pad
  X = kernel.get_shape()[1] + 2 * pad

  channels = kernel.get_shape()[2]

  # put NumKernels to the 1st dimension
  x = tf.transpose(x, (3, 0, 1, 2))
  # organize grid on Y axis
  x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

  # switch X and Y axes
  x = tf.transpose(x, (0, 2, 1, 3))
  # organize grid on X axis
  x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

  # back to normal order (not combining with the next step for clarity)
  x = tf.transpose(x, (2, 1, 3, 0))

  # to tf.image_summary order [batch_size_per_gpu, height, width, channels],
  #   where in this case batch_size_per_gpu == 1
  x = tf.transpose(x, (3, 0, 1, 2))

  # scaling to [0, 255] is not necessary for tensorboard
  return x

def gabor_regularization(tensor):
  "Input is a list of Complex Tensors"
  def euc_distance_matrix(size_y, size_x):
    # access first 2 indices to get the matrix of distance to location at those indices
    mat = np.zeros(shape=[size_y, size_x, size_y, size_x], dtype=np.float32)
    for i in range(size_y):
      for j in range(size_x):
        y = np.arange(size_y, dtype=np.float32) - i
        x = np.arange(size_x, dtype=np.float32) - j
        x, y = np.meshgrid(x, y)
        temp = np.sqrt(y ** 2 + x ** 2)
        mat[i, j] = temp
    mat = tf.constant(value=mat, shape=[size_y, size_x, size_y, size_x], dtype=tf.float32, verify_shape=True)
    return mat

  def center_of_mass(array):
    size_y = array.shape[0]
    size_x = array.shape[1]
    # print(array.shape)
    y_proj = np.mean(array, axis=1)
    # print(y_proj.shape)
    x_proj = np.mean(array, axis=0)
    x = np.arange(size_x, dtype=np.float32)
    y = np.arange(size_y, dtype=np.float32)

    mx = np.sum(x_proj * x) / np.sum(x_proj)
    mx = np.round(mx).astype(np.int32)
    my = np.sum(y_proj * y) / np.sum(y_proj)
    my = np.round(my).astype(np.int32)
    return my, mx

  def loss_per_filter(tensor, euc_mat):
    fft = np.fft.rfft2(a=tensor) #input_tensor=tensor, name='reg_fft') #fftshift2(tf.spectral.fft2d(tensor, 'reg_fft'))
    fft = np.abs(fft)
    centers = center_of_mass(array=fft)
    dist_mat = euc_mat[centers]
    weighted_dist = fft * dist_mat
    return np.mean(weighted_dist, dtype=np.float32)

  size_y = tensor.get_shape().as_list()[0]
  euc_mat = euc_distance_matrix(size_y=size_y, size_x=size_y // 2 + 1)
  list_of_tensors = tf.concat(tf.unstack(tensor, axis=3), axis=2)
  list_of_tensors = tf.unstack(list_of_tensors, axis=2)
  total_loss = 0
  for tensor in list_of_tensors:
    per_tensor_loss = tf.py_func(func=loss_per_filter, inp=[tensor, euc_mat], Tout=tf.float32, stateful=True,
                                 name='py_func_calculate_reg_per_tensor')
    total_loss += per_tensor_loss
  total_loss = total_loss / len(list_of_tensors)
  # print(total_loss.shape)
  total_loss.set_shape([])
  return total_loss

def rotate_im_to_re(im_weights, shape):
  # shape = [conv_filter_size, conv_filter_size, num_input_channels, num_filters]
  with tf.name_scope('rotate_im_to_re'):
    conv_filter_size = shape[0]
    num_input_channels = shape[2]
    num_filters = shape[3]
    minor_diag_np = np.tile(np.diag(np.ones(conv_filter_size))[::-1], [num_filters, num_input_channels, 1, 1])
    minor_diag = tf.constant(value=minor_diag_np, dtype=tf.float32, shape=shape[::-1], verify_shape=True,
                             name='minor_diag_ones_mat')
    re_weights_temp = tf.transpose(im_weights)
    re_weights_temp = tf.matmul(re_weights_temp, minor_diag)
    re_weights_temp = tf.transpose(re_weights_temp, perm=[2, 3, 1, 0])
    return re_weights_temp


""" Weight Generation"""
def create_weights_uniform(shape, name):
  with tf.name_scope(name):
    return tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                           shape=shape, name=name + '_w_uniform')


def create_biases(size):
  return tf.Variable(tf.constant(0.05, shape=[size]), name="b")


def create_conv2d_2d_gaussians(shape, name, trainable, sigma):
  def generate_2d_gaussian_bank(shape, sigma):
    # shape = [conv_filter_size, conv_filter_size, num_input_channels, num_filters]
    size = shape[0]
    num_input_channels = shape[2]
    num_filters = shape[3]
    gaussian = cv2.getGaussianKernel(ksize=size, sigma=sigma, ktype=cv2.CV_32F)
    gaussian = np.matmul(gaussian, gaussian.T)
    gaussian = gaussian / np.max(gaussian)
    weights = np.tile(gaussian, [num_filters, num_input_channels, 1, 1])
    weights = weights.T
    return weights

  with tf.name_scope(name):
    weights = tf.py_func(func=generate_2d_gaussian_bank, inp=[shape, sigma], Tout=[tf.float32], stateful=True,
                         name='py_func_generate_gabor_weights')
    weights = tf.reshape(weights, shape=shape, name='reshape_gabor_weights')
    weights = tf.Variable(initial_value=weights, trainable=trainable, name= '_w', dtype=tf.float32,
                               expected_shape=shape)
  return weights


""" Layers """
def learnable_po_conv2d_layer(name, image, num_input_channels, conv_filter_size, num_filters, is_training,
                              stride=1, conv=True, pool=True, relu=True, use_bn=True):
  with tf.name_scope(name):
    if conv:
      shape = [conv_filter_size, conv_filter_size, num_input_channels, num_filters]
      # tbprint('learnable_quadrature_conv2d_layer for image {}'.format(name))
      # im_weights = create_conv2d_2d_gabor_weights(shape=shape, name='im_weights', trainable=trainable, mode='imag',
      #                                             sigma_list=sigma_list, theta_list=theta_list, freq_list=freq_list)

      # random points as filters
      if not hp.gabor_reg:
        tbprint('quad using random initialization. Always Trainable')
        im_weights = create_weights_uniform(shape=shape, name='im_weights')
      else:
        # using fixed gaussian, trainable layer
        tbprint('quad using split filters. Always Trainable.')
        im_weights_gaussian = create_conv2d_2d_gaussians(shape=shape, name='im_weights_gaussian', trainable=False, sigma=2)
        im_weights_random = create_weights_uniform(shape=shape, name='im_weights_random')
        im_weights = im_weights_gaussian * im_weights_random

      re_weights_temp = rotate_im_to_re(im_weights=im_weights, shape=shape)
      # with tf.variable_scope('re_weights_scope', reuse=True):
      #   re_weights = tf.Variable(initial_value=re_weights_temp, trainable=False, validate_shape=True,
      #                                 expected_shape=shape, name='re_weights')
      re_weights = tf.get_variable(name='re_weights', shape=shape, trainable=False, dtype=tf.float32)
      update = tf.assign(re_weights, re_weights_temp, validate_shape=True, name='update_weights')

      with tf.get_default_graph().control_dependencies([update]):
        re_image = tf.nn.conv2d(input=image, filter=re_weights, strides=[1, stride, stride, 1], padding='SAME')
        im_image = tf.nn.conv2d(input=image, filter=im_weights, strides=[1, stride, stride, 1], padding='SAME')

      # re_weights_random = rotate_im_to_re(im_weights=im_weights_random, shape=shape)
      tf.get_collection('QUAD_REG')
      tf.add_to_collection(name='QUAD_REG', value=im_weights)


      tf.summary.histogram("re_weights", re_weights)
      tf.summary.histogram("im_weights", im_weights)
      tf.summary.histogram("re_conv_output", re_image)
      tf.summary.histogram("im_conv_output", im_image)

      if num_input_channels <= 4 and num_input_channels != 2:
        tf.summary.image('re_weights', put_kernels_on_grid(re_weights, 1), max_outputs=1)
        tf.summary.image('im_weights', put_kernels_on_grid(im_weights, 1), max_outputs=1)
      if num_input_channels > 4:
        tf.summary.image('re_weights', put_kernels_on_grid(re_weights[:, :, 0:4], 1), max_outputs=1)
        tf.summary.image('im_weights', put_kernels_on_grid(im_weights[:, :, 0:4], 1), max_outputs=1)

    if hp.complex_bn:
      tbprint('using complex_bn for layer {}'.format(name))
      with tf.name_scope('complex_bn'):
        with tf.variable_scope('complex_bn', reuse=tf.AUTO_REUSE):
          # complex_image = tf.concat([re_image, im_image], axis=-1)
          complex_image = keras.layers.Concatenate(axis=-1)([re_image, im_image])
          cbn = ComplexBatchNormalization()
          complex_image = cbn(complex_image)
        re_image, im_image = tf.split(complex_image, num_or_size_splits=2, axis=-1)
        assert re_image.get_shape().as_list() == im_image.get_shape().as_list(), 'split isnt correct'

    elif use_bn:
      re_image = tf.layers.batch_normalization(inputs=re_image, training=is_training, name=name + '_re_bn', renorm=True,
                                               fused=True)
      im_image = tf.layers.batch_normalization(inputs=im_image, training=is_training, name=name + '_im_bn', renorm=True,
                                               fused=True)
    else:
      biases = create_biases(num_filters)
      re_image += biases
      im_image += biases
      tf.summary.histogram("biases", biases)

    if relu:
      tf.summary.histogram("re_activations_pre_relu", re_image)
      tf.summary.histogram("im_activations_pre_relu", im_image)
      re_image = tf.nn.relu(re_image)
      im_image = tf.nn.relu(im_image)
      tf.summary.histogram("re_activations", re_image)
      tf.summary.histogram("im_activations", im_image)

      # """modReLu"""
      # tbprint('using modRelu for layer {}'.format(name))
      #
      # def modrelu(real, imag, b):
      #   with tf.name_scope('modrelu'):
      #     z_norm = tf.sqrt(tf.square(real) + tf.square(imag)) + 0.00001
      #     step1 = z_norm + b
      #     step2 = tf.complex(tf.nn.relu(step1), tf.zeros_like(z_norm))
      #     step3 = tf.complex(real, imag) / tf.complex(z_norm, tf.zeros_like(z_norm))
      #   return tf.multiply(step3, step2)
      #
      # tf.summary.histogram("re_activations_pre_relu", re_image)
      # tf.summary.histogram("im_activations_pre_relu", im_image)
      # modrelu_bias = tf.get_variable(name='modrelu_bias', shape=[num_filters],
      #                                initializer=tf.constant_initializer(value=0.0))
      # comp = modrelu(re_image, im_image, modrelu_bias)
      # re_image = tf.real(comp)
      # im_image = tf.imag(comp)
      # tf.summary.histogram('modrelu_bias', modrelu_bias)
      # tf.summary.histogram("re_activations", re_image)
      # tf.summary.histogram("im_activations", im_image)

      # """ zReLU """
      # tbprint('using zReLU for layer {}'.format(name))
      #
      # def zReLU(real, imag):
      #   zero = tf.constant(0.0, dtype=tf.float32)
      #   real_positive = tf.cast(tf.greater(real, zero), tf.float32)
      #   imag_positive = tf.cast(tf.greater(imag, zero), tf.float32)
      #   all_positive = real_positive * imag_positive
      #   return real * all_positive, imag * all_positive
      #
      # tf.summary.histogram("re_activations_pre_relu", re_image)
      # tf.summary.histogram("im_activations_pre_relu", im_image)
      # re_image, im_image = zReLU(re_image, im_image)
      # tf.summary.histogram("re_activations", re_image)
      # tf.summary.histogram("im_activations", im_image)

    if pool:
      re_image = tf.nn.max_pool(value=re_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
      im_image = tf.nn.max_pool(value=im_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    return re_image, im_image


def finetunable_po_conv2d_layer(name, image, pretrained_weights, num_input_channels, conv_filter_size, num_filters, is_training,
                                stride=1, conv=True, pool=True, relu=True, use_bn=True, padding='SAME'):
  with tf.name_scope(name):
    if conv:
      shape = [conv_filter_size, conv_filter_size, num_input_channels, num_filters]
      tbprint('finetunable quad using split filters. Always Trainable.')

      assert (pretrained_weights.get_shape().as_list()[2] == 3), 'pretrained weights dimension isnt RGB'
      pretrained_weights = tf.convert_to_tensor(pretrained_weights, dtype=tf.float32)

      # if mode is RGB
      if not hp.image_mode:
        pretrained_weights = tf.transpose(pretrained_weights, [0, 1, 3, 2])
        # pretrained_weights = tf.image.rgb_to_grayscale(pretrained_weights)
        pretrained_weights = tf.reduce_mean(pretrained_weights, axis=3, keep_dims=True)
        pretrained_weights = tf.transpose(pretrained_weights, [0, 1, 3, 2])

      # repeat to match input dimensions
      pretrained_weights = tf.tile(pretrained_weights, [1, 1, hp.stack_depth, 1])
      # # renormalize weights
      # pretrained_weights = pretrained_weights / hp.stack_depth

      im_weights = tf.Variable(initial_value=pretrained_weights, trainable=True, name='im_weights', dtype=tf.float32,
                               expected_shape=shape)
      re_weights_temp = rotate_im_to_re(im_weights=im_weights, shape=shape)
      re_weights = tf.get_variable(name='re_weights', shape=shape, trainable=False, dtype=tf.float32)
      update = tf.assign(re_weights, re_weights_temp, validate_shape=True, name='update_weights')

      with tf.get_default_graph().control_dependencies([update]):
        re_image = tf.nn.conv2d(input=image, filter=re_weights, strides=[1, stride, stride, 1], padding='SAME')
        im_image = tf.nn.conv2d(input=image, filter=im_weights, strides=[1, stride, stride, 1], padding='SAME')

      tf.get_collection('QUAD_REG')
      tf.add_to_collection(name='QUAD_REG', value=im_weights)


      # if num_input_channels <= 4:
      #   tf.summary.image('re_weights', put_kernels_on_grid(re_weights, 1), max_outputs=1)
      #   tf.summary.image('im_weights', put_kernels_on_grid(im_weights, 1), max_outputs=1)
      # if num_input_channels > 4:
      #   tf.summary.image('re_weights', put_kernels_on_grid(re_weights[:,:,4], 1), max_outputs=1)
      #   tf.summary.image('im_weights', put_kernels_on_grid(im_weights[:,:,4], 1), max_outputs=1)

    if use_bn:
      re_image = tf.layers.batch_normalization(inputs=re_image, training=is_training, name=name + '_re_bn', renorm=True,
                                               fused=True)
      im_image = tf.layers.batch_normalization(inputs=im_image, training=is_training, name=name + '_im_bn', renorm=True,
                                               fused=True)
    else:
      biases = create_biases(num_filters)
      re_image += biases
      im_image += biases
      tf.summary.histogram("biases", biases)

    if relu:
      re_image = tf.nn.relu(re_image)
      im_image = tf.nn.relu(im_image)
      tf.summary.histogram("re_activations", re_image)
      tf.summary.histogram("im_activations", im_image)

    if pool:
      re_image = tf.nn.max_pool(value=re_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)
      im_image = tf.nn.max_pool(value=im_image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding=padding)

    return re_image, im_image

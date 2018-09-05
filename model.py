import os
import sys

import numpy as np
import tensorflow as tf

from utils import *

EMOTIONS = ['angry', 'disgusted', 'fearful', 'happy', 'sad', 'surprised', 'neutral']

classes = len(EMOTIONS)

def inference2_google(images):
    x_image = tf.reshape(images, [-1, 48, 48, 1])

    with tf.variable_scope('conv1') as scope:
        conv1 = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
            kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        layer1_image1 = conv1[0:1, :, :, 0:16]
        layer1_image1 = tf.transpose(layer1_image1, perm=[3, 1, 2, 0])
        tf.summary.image('layer1_image1', layer1_image1, max_outputs=16)

    with tf.variable_scope('pool1') as scope:
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], strides=2)
        tf.summary.histogram('pool1', pool1)

    with tf.variable_scope('conv2') as scope:
        conv2_b1 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))


        conv2_b2 = tf.layers.conv2d(
            inputs=pool1,
            filters=32,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_b2 = tf.layers.conv2d(
            inputs=conv2_b2,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

        conv2_b3 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_b3 = tf.layers.conv2d(
            inputs=conv2_b3,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

        conv2 = tf.concat([conv2_b1, conv2_b2, conv2_b3],3)
        print(conv1.shape)
        print(conv2.shape)

        layer1_image2 = conv2[0:1, :, :, 0:80]
        layer1_image2 = tf.transpose(layer1_image2, perm=[3, 1, 2, 0])
        tf.summary.image('layer1_image2', layer1_image2, max_outputs=80)

    with tf.variable_scope('pool2') as scope:
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], strides=2)
        tf.summary.histogram('pool2', pool2)


    shape = int(np.prod(pool2.get_shape()[1:]))
    flatten = tf.reshape(pool2, [-1, shape])
    with tf.variable_scope('dense1') as scope:
        dense1 = tf.layers.dense(inputs=flatten,
                                units=1280,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        tf.summary.histogram('dense1', dense1)

    with tf.variable_scope('dense2') as scope:
        dense2 = tf.layers.dense(inputs=dense1,
                                units=256,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        tf.summary.histogram('dense2', dense2)

    with tf.variable_scope('logits'):
        logits = tf.layers.dense(inputs=dense2,
                                units=classes,
                                activation=None,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        tf.summary.histogram('logits', logits)
    return logits

def vgg19(images):
    x_image = tf.reshape(images, [-1, 48, 48, 1])

    with tf.variable_scope('conv1_1') as scope:
        conv1_1 = tf.layers.conv2d(
            inputs=x_image,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('conv1_2') as scope:
        conv1_2 = tf.layers.conv2d(
            inputs=conv1_1,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('pool1') as scope:
        pool1 = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv2_1') as scope:
        conv2_1 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('conv2_2') as scope:
        conv2_2 = tf.layers.conv2d(
            inputs=conv2_1,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('pool2') as scope:
        pool2 = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv3_1') as scope:
        conv3_1 = tf.layers.conv2d(
            inputs=pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('conv3_2') as scope:
        conv3_2 = tf.layers.conv2d(
            inputs=conv3_1,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('conv3_3') as scope:
        conv3_3 = tf.layers.conv2d(
            inputs=conv3_2,
            filters=128,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('pool3') as scope:
        pool3 = tf.layers.max_pooling2d(inputs=conv3_3, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv4_1') as scope:
        conv4_1 = tf.layers.conv2d(
            inputs=pool3,
            filters=512,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('conv4_2') as scope:
        conv4_2 = tf.layers.conv2d(
            inputs=conv4_1,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('conv4_3') as scope:
        conv4_3 = tf.layers.conv2d(
            inputs=conv4_2,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('conv4_4') as scope:
        conv4_4 = tf.layers.conv2d(
            inputs=conv4_3,
            filters=1024,
            kernel_size=[1, 1],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('pool4') as scope:
        pool4 = tf.layers.max_pooling2d(inputs=conv4_4, pool_size=[2, 2], strides=2)

    with tf.variable_scope('conv5_1') as scope:
        conv5_1 = tf.layers.conv2d(
            inputs=pool4,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('conv5_2') as scope:
        conv5_2 = tf.layers.conv2d(
            inputs=conv5_1,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('conv5_3') as scope:
        conv5_3 = tf.layers.conv2d(
            inputs=conv5_2,
            filters=512,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.1))

    with tf.variable_scope('pool5') as scope:
        pool5 = tf.layers.max_pooling2d(inputs=conv5_3, pool_size=[2, 2], strides=2)


    shape = int(np.prod(pool5.get_shape()[1:]))
    flatten = tf.reshape(pool5, [-1, shape])

    with tf.variable_scope('dense1') as scope:
        dense1 = tf.layers.dense(inputs=flatten,
                                units=1024,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))

    with tf.variable_scope('dense1_drop') as scope:
        dense1_drop = tf.layers.dropout(inputs=dense1, rate=0.5)

    with tf.variable_scope('dense2') as scope:
        dense2 = tf.layers.dense(inputs=dense1_drop,
                                units=512,
                                activation=tf.nn.relu,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))

    with tf.variable_scope('dense2_drop') as scope:
        dense2_drop = tf.layers.dropout(inputs=dense2, rate=0.5)

    with tf.variable_scope('logits'):
        logits = tf.layers.dense(inputs=dense2_drop,
                                units=classes,
                                activation=None,
                                kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
    return logits

def deepnn(x):
  x_image = tf.reshape(x, [-1, 48, 48, 1])
  # conv1
  W_conv1 = weight_variables([5, 5, 1, 64])
  b_conv1 = bias_variable([64])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  # pool1
  h_pool1 = maxpool(h_conv1)
  # norm1
  norm1 = tf.nn.lrn(h_pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

  # conv2
  W_conv2 = weight_variables([3, 3, 64, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  norm2 = tf.nn.lrn(h_conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
  h_pool2 = maxpool(norm2)

  # Fully connected layer
  W_fc1 = weight_variables([12 * 12 * 64, 384])
  b_fc1 = bias_variable([384])
  h_conv3_flat = tf.reshape(h_pool2, [-1, 12 * 12 * 64])
  h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

  # Fully connected layer
  W_fc2 = weight_variables([384, 192])
  b_fc2 = bias_variable([192])
  h_fc2 = tf.matmul(h_fc1, W_fc2) + b_fc2

  # linear
  W_fc3 = weight_variables([192, 7])
  b_fc3 = bias_variable([7])
  y_conv = tf.add(tf.matmul(h_fc2, W_fc3), b_fc3)

  return y_conv


def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def maxpool(x):
  return tf.nn.max_pool(x, ksize=[1, 3, 3, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variables(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def train_model(train_data):
  fer2013 = input_data(train_data)
  max_train_steps = 300001

  x = tf.placeholder(tf.float32, [None, 2304])
  y_ = tf.placeholder(tf.float32, [None, 7])

  y_conv = vgg19(x)

  cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
  global_step = tf.Variable(0, name='global_step', trainable=False)
  learning_rate = tf.train.exponential_decay(1e-3, global_step, decay_steps=5000, decay_rate=0.95,
                                             staircase=True)
  train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cross_entropy,global_step = global_step)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    for step in range(max_train_steps):
      batch = fer2013.train.next_batch(100)
      if step % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1]})
        print('step %d,lr%f, training accuracy %g' % (step, learning_rate.eval(), train_accuracy))
      train_step.run(feed_dict={x: batch[0], y_: batch[1]})

      if step % 1000 == 0:
        saver.save(sess, './models/emotion_model', global_step=step + 1)
      if step % 1000 == 0:
        print('*Test accuracy %g' % accuracy.eval(feed_dict={
          x: fer2013.validation.images, y_: fer2013.validation.labels}))


def predict(image=[[0.1] * 2304]):
  x = tf.placeholder(tf.float32, [None, 2304])
  y_conv = deepnn(x)

  # init = tf.global_variables_initializer()
  saver = tf.train.Saver()
  probs = tf.nn.softmax(y_conv)
  y_ = tf.argmax(probs)

  with tf.Session() as sess:
    # assert os.path.exists('/tmp/models/emotion_model')
    ckpt = tf.train.get_checkpoint_state('./models')
    print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print('Restore ssss')
    return sess.run(probs, feed_dict={x: image})


def image_to_tensor(image):
  tensor = np.asarray(image).reshape(-1, 2304) * 1 / 255.0
  return tensor


def valid_model(modelPath, validFile):
  x = tf.placeholder(tf.float32, [None, 2304])
  y_conv = deepnn(x)
  probs = tf.nn.softmax(y_conv)

  saver = tf.train.Saver()
  ckpt = tf.train.get_checkpoint_state(modelPath)

  with tf.Session() as sess:
    print(ckpt.model_checkpoint_path)
    if ckpt and ckpt.model_checkpoint_path:
      saver.restore(sess, ckpt.model_checkpoint_path)
      print('Restore model sucsses!!')

    files = os.listdir(validFile)

    for file in files:
      if file.endswith('.jpg'):
        image_file = os.path.join(validFile, file)
        image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
        tensor = image_to_tensor(image)
        result = sess.run(probs, feed_dict={x: tensor})
        print(file, EMOTIONS[result.argmax()])

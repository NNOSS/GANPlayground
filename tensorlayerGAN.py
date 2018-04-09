from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *

class discriminator:
    def __init__(self,inputSize,convolutions,fullyconnected,output):
        self.inputSize = inputSize
        self.convolutions= [1] + convolutions
        self.fullyconnected = fullyconnected
        self.sess = tf.InteractiveSession()

        self.sess.run(tf.global_variables_initializer())

    def createGenerator(self,inputSize,convolutions,fullyconnected,output):
        # Generator Net
        self.Z = tf.placeholder(tf.float32, shape=[None, inputSize], name='Z')
        numPools = sum([1 if i < 0 for i in convoltions])
        xs, ys = output[0]/(2^numPools),output[1]/(2^numPools)
        sizeDeconv = xs * ys * convoltions[0]

        fcG = DenseLayer(self.Z, fullyconnected, act = tf.nn.relu, name = 'fc2')
        deconveInputFlat = DenseLayer(fcG, sizeDeconv, act = tf.nn.relu, name = 'fdeconv')
        deconveInput = ReshapeLayer(deconveInputFlat, (-1, xs, ys, convoltions[0]), name = 'unflatten')

        conv3 = Conv2d(shape3, 8, (3, 3), act=tf.nn.relu, name='conv3_2')
        up2 = DeConv2d(conv3, 8, (3, 3), (nx/2, ny/2), (2, 2), name='deconv2')
        convoltions.append(1)
        convVals = [deconveInput]

        for i,v in enumerate(convoltions):
            if i < len(convoltions)-1:
                if convolutions[i] < 0:
                    convolutions[i] *= -1
                    xs *= 2
                    ys *= 2
                convVals.append(DeConv2d(convVals[i],convoltions[i+1], (5, 5), (xs,ys), (2, 2), name='deconv%s'%(i)))

        self.genOutput = convVals[-1]



    def createDiscriminator(self):
        flatSize = inputSize[0]*inputSize[1]*inputSize[2]
        self.x = tf.placeholder(tf.float32, shape=[None, flatSize])
        x_image = tf.reshape(x, [-1,inputSize[0],inputSize[1],inputSize[2]])
        X_255 = tf.image.convert_image_dtype (x_image, dtype=tf.uint8)
        #X_t = tf.transpose (X_255, [3, 0, 1, 2])
        tf.summary.image('input', X_255, max_outputs = 3)
        inputs = InputLayer(x_image, name='disc_inputs')
        self.y_ = tf.placeholder(tf.float32, shape=[None, output])
        convVals = [inputs]
        for i,v in enumerate(convoltions):
            if i < len(convoltions)-1:#if it is negative, that means we pool on this step
                pool=False
                if self.convolutions[i+1] < 0:
                    self.convolutions[i+1] *= -1
                    pool = True
                with tf.variable_scope("u_net", reuse=None):
                    conv1 = Conv2d(inputs, self.convolutions[i+1], (5, 5), act=tf.nn.relu, name='conv1_%s'%(i))
                if pool:
                    convVals.append(MaxPool2d(conv1, (2, 2), name='pool%s'%(i)))
                else:
                    convVals.append(conv1)
            else:
                _,l,w,d = convVals[-1].shape
                flat3 = FlattenLayer(convVals[-1], name = 'flatten')
                hid3 = DenseLayer(flat3, fullyconnected, act = tf.nn.relu, name = 'fcl')
                self.y_conv = DenseLayer(hid, output, act = tf.nn.relu, name = 'hidden_encode')

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar("loss", cross_entropy);


    def train(iterations,batch = 50):
        for i in range(iterations):
          getbatch()
          if i%100 == 0:
            train_accuracy = self.accuracy.eval(feed_dict={
                self.x:batch[0], self.y_: batch[1], self.keep_prob: 1.0})
            print("step %d, training accuracy %g"%(i, train_accuracy))
          self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: 0.5})

        print("test accuracy %g"%self.accuracy.eval(feed_dict={
            self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0}))



    def variable_summaries(var):
      """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)



    merged = tf.summary.merge_all()
    sum_writer = tf.summary.FileWriter("logs2", sess.graph)

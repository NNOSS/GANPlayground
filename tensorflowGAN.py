'''Source
https://wiseodd.github.io/techblog/2016/09/17/gan-tensorflow/'''
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
        self.Z = tf.placeholder(tf.float32, shape=[None, 100], name='Z')

        G_W1 = tf.Variable(xavier_init([100, 128]), name='G_W1')
        G_b1 = tf.Variable(tf.zeros(shape=[128]), name='G_b1')

        G_W2 = tf.Variable(xavier_init([128, 784]), name='G_W2')
        G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')

        theta_G = [G_W1, G_W2, G_b1, G_b2]
        def generator():
            G_h1 = tf.nn.relu(tf.matmul(z, G_W1) + G_b1)
            G_log_prob = tf.matmul(G_h1, G_W2) + G_b2
            G_prob = tf.nn.sigmoid(G_log_prob)

            return G_prob

        convVals = []
        convWeights = []
        convBias = []
        for i,v in enumerate(convoltions):
            if i > 1:#if it is negative, that means we pool on this step
                pool=False
                if self.convolutions[i+1] < 0:
                    self.convolutions[i+1] *= -1
                    pool = True
                convWeights.append(weight_variable([5, 5, self.convolutions[i], self.convolutions[i+1]]))
                convBias.append(bias_variable([self.convolutions[i+1]]))
                h_conv1 = tf.nn.relu(conv2d(convVals[i], convWeights[i]) + convBias[i])
                if pool:
                    convVals.append(max_pool_2x2(h_conv1))
                else:
                    convVals.append(h_conv1)
            else:
                W_fc1 = weight_variable([inputSize, fullyconnected])
                b_fc1 = bias_variable([fullyconnected])
                h_fc1 = tf.nn.relu(tf.matmul(self.Z, W_fc1) + b_fc1)

                W_fc2 = weight_variable([fullyconnected, output])
                b_fc2 = bias_variable([output])
                self.y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2

    def createDiscriminator(self):
        flatSize = inputSize[0]*inputSize[1]*inputSize[2]
        self.x = tf.placeholder(tf.float32, shape=[None, flatSize])
        x_image = tf.reshape(x, [-1,inputSize[0],inputSize[1],inputSize[2]])
        self.y_ = tf.placeholder(tf.float32, shape=[None, output])
        convVals = [x_image]
        convWeights = []
        convBias = []
        for i,v in enumerate(convoltions):
            if i < len(convoltions)-1:#if it is negative, that means we pool on this step
                pool=False
                if self.convolutions[i+1] < 0:
                    self.convolutions[i+1] *= -1
                    pool = True
                convWeights.append(weight_variable([5, 5, self.convolutions[i], self.convolutions[i+1]]))
                convBias.append(bias_variable([self.convolutions[i+1]]))
                h_conv1 = tf.nn.relu(conv2d(convVals[i], convWeights[i]) + convBias[i])
                if pool:
                    convVals.append(max_pool_2x2(h_conv1))
                else:
                    convVals.append(h_conv1)
            else:
                _,l,w,d = convVals[-1].shape
                W_fc1 = weight_variable([l*w*d, fullyconnected])
                b_fc1 = bias_variable([fullyconnected])
                h_pool2_flat = tf.reshape(convVals[-1], [-1, l*w*d])
                h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
                self.keep_prob = tf.placeholder(tf.float32)
                h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)
                W_fc2 = weight_variable([fullyconnected, output])
                b_fc2 = bias_variable([output])
                self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
            self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


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


    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def deconv2d(x,W,upPool):
        return tf.nn.conv2d_transpose(
    value=x,
    filter=w,
    output_shape=,
    padding='SAME',
)
        )

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')


merged = tf.summary.merge_all()
sum_writer = tf.summary.FileWriter("logs2", sess.graph)

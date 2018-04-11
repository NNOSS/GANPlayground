from __future__ import print_function

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import numpy as np
import matplotlib.pyplot as plt

class discriminator:
    def __init__(self,inputSize,convolutions,fullyconnected,output,restore = False,fileName =None):
        self.inputSize = inputSize
        self.convolutions= [1] + convolutions
        self.fullyconnected = fullyconnected
        self.output = output
        self.Zsize =30
        self.fileName = fileName
        self.sess = tf.InteractiveSession()

        self.Z = tf.placeholder(tf.float32, shape=[None, self.Zsize], name='Z')
        self.y_ = tf.placeholder(tf.float32, shape=[None, output])
        self.fake_y_ =tf.placeholder(tf.float32, shape=[None, output])
        self.gen_y_ = tf.placeholder(tf.float32, shape=[None, output])
        self.x = tf.placeholder(tf.float32, shape=[None, inputSize[0]*inputSize[1]*inputSize[2]])
        self.y_conv = self.createDiscriminator(self.x,inputSize,self.convolutions,fullyconnected,output)
        print(tf.get_variable_scope().reuse)


        self.convolutions = [1] + convolutions
        self.fake_input = self.createGenerator(Z = self.Z, inputSize = 30, convolutions= [-16, -32],fullyconnected = 512, output = [28, 28, 1])
        print(tf.get_variable_scope().reuse)
        self.fake_y_conv = self.createDiscriminator(self.fake_input,inputSize,self.convolutions,fullyconnected,output,reuse = True)


        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)) + tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.fake_y_, logits=self.fake_y_conv))
        self.train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.gen_y_, logits=self.fake_y_conv))
        self.gen_train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        self.saver = tf.train.Saver()
        if fileName is not None and restore:
            self.saver.restore(self.sess, self.fileName)
        self.sess.run(tf.global_variables_initializer())
        self.saver.save(self.sess, self.fileName)


    def createGenerator(self,Z, inputSize,convolutions,fullyconnected,output):
        # Generator Net
        inputs = InputLayer(Z, name='generator_inputs')
        numPools = sum([1 for i in convolutions if i < 0])
        print(numPools)
        xs, ys = output[0]/(2**numPools),output[1]/(2**numPools)
        sizeDeconv = xs * ys * abs(convolutions[0])

        fcG = DenseLayer(inputs, fullyconnected, act = tf.nn.relu, name = 'fc2')
        deconveInputFlat = DenseLayer(fcG, sizeDeconv, act = tf.nn.relu, name = 'fdeconv')
        deconveInput = ReshapeLayer(deconveInputFlat, (-1, xs, ys, abs(convolutions[0])), name = 'unflatten')

        convolutions.append(1)
        convVals = [deconveInput]

        for i,v in enumerate(convolutions):
            if i < len(convolutions)-1:
                if convolutions[i] < 0:
                    convolutions[i] *= -1
                    xs *= 2
                    ys *= 2
                    stride  = (2,2)
                else:
                    stride = (1,1)
                print('deconv%i'% i)
                convVals.append(DeConv2d(convVals[i],abs(convolutions[i+1]), (5, 5), (xs,ys), stride, name='deconv%i'%(i)))

        return FlattenLayer(convVals[-1]).outputs


    def createDiscriminator(self,x,inputSize,convolutions,fullyconnected,output,reuse=False):
         with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            flatSize = inputSize[0]*inputSize[1]*inputSize[2]
            x_image = tf.reshape(x, [-1,inputSize[0],inputSize[1],inputSize[2]])
            X_255 = tf.image.convert_image_dtype (x_image, dtype=tf.uint8)
            numPools = sum([1 for i in convolutions if i < 0])
            print(numPools)
            #X_t = tf.transpose (X_255, [3, 0, 1, 2])
            tf.summary.image('input', X_255, max_outputs = 3)
            inputs = InputLayer(x_image, name='disc_inputs')

            convVals = [inputs]
            for i,v in enumerate(convolutions):
                if i < len(convolutions)-1:#if it is negative, that means we pool on this step
                    pool=False
                    if convolutions[i+1] < 0:
                        convolutions[i+1] *= -1
                        pool = True

                    conv1 = Conv2d(inputs, convolutions[i+1], (5, 5), act=tf.nn.relu, name='conv1_%s'%(i))
                    if pool:
                        convVals.append(MaxPool2d(conv1, (2, 2), name='pool%s'%(i)))
                    else:
                        convVals.append(conv1)
                else:
                    print(convVals[-1])
                    l,w,d = inputSize[0]/(2**numPools),inputSize[1]/(2**numPools),convolutions[-1]
                    flat3 = FlattenLayer(convVals[-1], name = 'flatten')
                    hid3 = DenseLayer(flat3, fullyconnected, act = tf.nn.relu, name = 'fcl')
                    y_conv = DenseLayer(hid3, output, act = tf.nn.relu, name = 'hidden_encode').outputs


            return y_conv

    def train(self,iterations,batchLen = 50):
        zt = np.random.normal(loc= .5, scale = .5, size = [batchLen,self.Zsize])
        for i in range(iterations):
            batch = self.getbatch(batchLen)
            if i%200 == 0:
                train_accuracy = self.accuracy.eval(feed_dict={
                    self.x:batch[0], self.y_: batch[1]})
                print("step %d, training accuracy %g"%(i, train_accuracy))

                # saver.save(sess, './Model2/Mnist-Encoding', global_step=save_step, write_meta_graph=False)
                if i % 200 ==0:
                    if self.fileName is not None:
                        self.saver.save(self.sess, self.fileName)
                    image = self.sess.run(self.fake_input, feed_dict={self.Z : zt})
                    image = np.reshape(image[6], [28,28])
                    # print(image)
                    plot = plt.imshow(image)
                    plt.show()

            z = np.random.normal(loc= .5, scale = .5, size = [batchLen,self.Zsize])
            fake_batch = np.concatenate((np.ones((batchLen,1)),np.zeros((batchLen,1))), axis = 1)
            gen_batch = np.concatenate((np.zeros((batchLen,1)),np.ones((batchLen,1))), axis = 1)
            # print(fake_batch.shape)
            self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1],self.Z: z, self.fake_y_: fake_batch})
            self.gen_train_step.run(feed_dict={self.Z: z, self.gen_y_: gen_batch})
        # print("test accuracy %g"%self.accuracy.eval(feed_dict={
        #     self.x: mnist.test.images, self.y_: mnist.test.labels}))

    def getbatch(self,batchLen):
        batch = mnist.train.next_batch(batchLen)
        newBatch = np.concatenate((np.zeros((batchLen,1)),np.ones((batchLen,1))), axis = 1)
        return batch[0], newBatch

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

myDiscriminator = discriminator(inputSize = [28,28,1],convolutions = [-16,-32], fullyconnected = 512, output = 2, fileName = './Model/model.ckpt',restore = True)
myDiscriminator.train(10000)

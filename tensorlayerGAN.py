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
        self.numClasses =10
        self.fileName = fileName
        self.sess = tf.InteractiveSession()

        self.Z = tf.placeholder(tf.float32, shape=[None, self.Zsize], name='Z')
        self.classes = tf.placeholder(tf.float32, shape=[None, self.numClasses], name='class_inputs')
        self.y_ = tf.placeholder(tf.float32, shape=[None, output])
        self.fake_y_ =tf.placeholder(tf.float32, shape=[None, output])
        self.gen_y_ = tf.placeholder(tf.float32, shape=[None, output])
        self.x = tf.placeholder(tf.float32, shape=[None, inputSize[0]*inputSize[1]*inputSize[2]])
        self.y_conv = self.createDiscriminator(self.x,self.classes,inputSize,self.convolutions,fullyconnected,output)
        print(tf.get_variable_scope().reuse)


        self.convolutions = [1] + convolutions
        self.fake_input = self.createGenerator(Z = self.Z, classes = self.classes,inputSize = self.Zsize, convolutions= [-16,16,-16,16],fullyconnected = 128, output = [28, 28, 1])
        print(tf.get_variable_scope().reuse)
        self.fake_y_conv = self.createDiscriminator(self.fake_input,self.classes,inputSize,self.convolutions,fullyconnected,output,reuse = True)


        t_vars = tf.trainable_variables()
        print(t_vars)

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        print(self.d_vars)
        self.gen_vars = [var for var in t_vars if 'gen_' in var.name]
        print(self.gen_vars)
        d_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)) + tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.fake_y_, logits=self.fake_y_conv))
        self.train_step = tf.train.AdamOptimizer(2e-4,beta1=.5).minimize(d_cross_entropy, var_list = self.d_vars)
        correct_prediction = tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        gen_cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=self.gen_y_, logits=self.fake_y_conv))
        self.gen_train_step = tf.train.AdamOptimizer(2e-4,beta1=.5).minimize(gen_cross_entropy,var_list = self.gen_vars)

        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())



        if fileName is not None and restore:
            self.saver.restore(self.sess, self.fileName)
        else:

            self.saver.save(self.sess, self.fileName)


    def createGenerator(self,Z, classes, inputSize,convolutions,fullyconnected,output):
        # Generator Net
        inputs = InputLayer(Z, name='gen_inputs')
        inputClass =InputLayer(classes, name='gen_class_inputs_z')
        concatz = ConcatLayer([inputs, inputClass], 1, name ='gen_concat_layer_z')
        numPools = sum([1 for i in convolutions if i < 0])
        print(numPools)
        xs, ys = output[0]/(2**numPools),output[1]/(2**numPools)
        sizeDeconv = xs * ys * abs(convolutions[0])

        fcG = DenseLayer(concatz, fullyconnected, act = tf.nn.relu, name = 'gen_fc')
        deconveInputFlat = DenseLayer(fcG, sizeDeconv, act = tf.nn.relu, name = 'gen_fdeconv')
        deconveInput = ReshapeLayer(deconveInputFlat, (-1, xs, ys, abs(convolutions[0])), name = 'gen_unflatten')

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
                if i == len(convolutions)-2:
                    convVals.append(DeConv2d(convVals[i],abs(convolutions[i+1]), (5, 5), (xs,ys), stride, act=tf.nn.tanh,name='gen_deconv%i'%(i)))

                else:
                    convVals.append(DeConv2d(convVals[i],abs(convolutions[i+1]), (5, 5), (xs,ys), stride, act=tf.nn.relu,name='gen_deconv%i'%(i)))


        return FlattenLayer(convVals[-1]).outputs


    def createDiscriminator(self,x,classes,inputSize,convolutions,fullyconnected,output,reuse=False):
         with tf.variable_scope("d_discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            flatSize = inputSize[0]*inputSize[1]*inputSize[2]
            x_image = tf.reshape(x, [-1,inputSize[0],inputSize[1],inputSize[2]])
            X_255 = tf.image.convert_image_dtype (x_image, dtype=tf.uint8)
            numPools = sum([1 for i in convolutions if i < 0])
            print(numPools)
            #X_t = tf.transpose (X_255, [3, 0, 1, 2])
            tf.summary.image('d_input', X_255, max_outputs = 3)
            inputs = InputLayer(x_image, name='d_disc_inputs')

            convVals = [inputs]
            for i,v in enumerate(convolutions):
                if i < len(convolutions)-1:#if it is negative, that means we pool on this step
                    pool=False
                    if convolutions[i+1] < 0:
                        convolutions[i+1] *= -1
                        pool = True

                    if pool:
                        conv1 = Conv2d(inputs, convolutions[i+1], (5, 5),strides = (2,2), act=tf.nn.relu, name='d_conv1_%s'%(i))
                    else:
                        conv1 = Conv2d(inputs, convolutions[i+1], (5, 5),strides = (1,1), act=tf.nn.relu, name='d_conv1_%s'%(i))
                else:
                    print(convVals[-1])
                    l,w,d = inputSize[0]/(2**numPools),inputSize[1]/(2**numPools),convolutions[-1]
                    flat3 = FlattenLayer(convVals[-1], name = 'd_flatten')
                    inputClass =InputLayer(classes, name='d_class_inputs')
                    concat = ConcatLayer([flat3, inputClass], 1, name ='d_concat_layer')
                    hid3 = DenseLayer(concat, fullyconnected, act = tf.nn.relu, name = 'd_fcl')
                    y_conv = DenseLayer(hid3, output, act = tf.nn.relu, name = 'd_hidden_encode').outputs


            return y_conv

    def train(self,iterations,batchLen = 120):
        random_classes = np.random.randint(0, self.numClasses, size = [batchLen])
        print(random_classes[0])
        positions = np.arange(batchLen)
        onehot = np.zeros((batchLen,self.numClasses))
        onehot[positions, random_classes] = 1
        # zt = np.concatenate((np.random.normal(loc= .5, scale = .5, size = [batchLen,self.Zsize]),
        # onehot),axis = 1)
        zt = np.random.normal(loc= .5, scale = .5, size = [batchLen,self.Zsize])
        classes = onehot
        for i in range(iterations):
            batch = self.getbatch(batchLen)

            # z = np.concatenate((np.random.normal(loc= .5, scale = .5, size = [batchLen,self.Zsize]),batch[2]),axis = 1)
            z = np.random.normal(loc= .5, scale = .5, size = [batchLen,self.Zsize])
            fake_batch = np.concatenate((np.ones((batchLen,1)),np.zeros((batchLen,1))), axis = 1)
            gen_batch = np.concatenate((np.zeros((batchLen,1)),np.ones((batchLen,1))), axis = 1)
            # print(fake_batch.shape)

            if i%20 == 0:
                train_accuracy = self.accuracy.eval(feed_dict={self.x: batch[0], self.y_: batch[1],self.Z: z, self.fake_y_: fake_batch, self.classes: batch[2]})
                print("step %d, training accuracy %g"%(i, train_accuracy))

                # saver.save(sess, './Model2/Mnist-Encoding', global_step=save_step, write_meta_graph=False)
                if i % 100 ==0:
                    if self.fileName is not None:
                        self.saver.save(self.sess, self.fileName)
                    image = self.sess.run(self.fake_input, feed_dict={self.Z : zt, self.classes: classes})
                    image = np.reshape(image[0], [28,28])
                    # realImage = np.reshape(batch[0][0], [28,28])
                    # print(image)
                    # print(batch[0][0])
                    # print(image)
                    plot = plt.imshow(image)
                    plt.show()

            self.train_step.run(feed_dict={self.x: batch[0], self.y_: batch[1],self.Z: z, self.fake_y_: fake_batch, self.classes: batch[2]})
            self.gen_train_step.run(feed_dict={self.Z: z, self.gen_y_: gen_batch, self.classes: batch[2]})
        # print("test accuracy %g"%self.accuracy.eval(feed_dict={
        #     self.x: mnist.test.images, self.y_: mnist.test.labels}))

    def getbatch(self,batchLen):
        batch = mnist.train.next_batch(batchLen)
        newBatch = np.concatenate((np.zeros((batchLen,1)),np.ones((batchLen,1))), axis = 1)
        return batch[0], newBatch, batch[1]

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

myDiscriminator = discriminator(inputSize = [28,28,1],convolutions = [-32,-64], fullyconnected = 128, output = 2, fileName = './../GANModel/model.ckpt',restore = False)
myDiscriminator.train(100000)

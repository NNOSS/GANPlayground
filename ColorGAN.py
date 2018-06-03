from __future__ import print_function

# Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
import tensorlayer as tl
import numpy as np
from tensorlayer.layers import *
import numpy as np
import matplotlib.pyplot as plt
import importImg
import saveMovie
import imageio

# importCIFAR.maybe_download_and_extract()
folder = 'celebA/*.jpg'
restore = False #whether or not to restor the file from a source
get_video = False
whenSave = 500
iterations = 1000000

classes = None
convolutions = [-128, -128, -128, -128, 128]
fullyconnected = 1024

z_learning_rate = 1e-4
learning_rate = 1e-4
batch_size = 64

model_filepath = './Models/Faces/model.ckpt' #filepaths to model and summaries
summary_filepath = './Models/Faces/Summaries/'
box = (150,150)
inputSize = [150,150,3]
outputs = 1

outputsFake = 15
outputsReal = 3
reconfig_outputs = 3
tbWhenSavePicture = 15

whenAddPicture = 10
whenSaveMovie = 500
imageFilePath = './Models/Faces/refineMovie.gif'


# model_filepath = './../thisworks/model.ckpt'
# summary_filepath = './../thisworks/Summaries/'

class GAN:
    def __init__(self,inputSize,convolutions,fullyconnected,output,restore = False,fileName =None, classes = None):
        self.defineVariables(inputSize,convolutions,fullyconnected,output,restore,fileName, classes)
        self.defineModels()
        self.defineLearner()
        self.generateZMaps()
        t_vars = tf.trainable_variables()

        #self.get_vars = [var for var in t_vars if 'z_' not in var.name] #find trainable discriminator variable
        #for var in self.get_vars:
        #     print(var.name)

        self.saver = tf.train.Saver()#saving and tensorboard
        self.sess.run(tf.global_variables_initializer())
        self.train_writer = tf.summary.FileWriter(summary_filepath,
                                      self.sess.graph)
        if fileName is not None and restore:
            self.saver.restore(self.sess, self.fileName)
        else:
            self.saver.save(self.sess, self.fileName)

        #self.saver = tf.train.Saver()


        # self.sess.run(tf.global_variables_initializer())

    def defineLearner(self):
        t_vars = tf.trainable_variables()
        print(t_vars)

        self.d_vars = [var for var in t_vars if 'd_' in var.name] #find trainable discriminator variable
        for var in self.d_vars:
            print(var.name)
        self.gen_vars = [var for var in t_vars if 'gen_' in var.name] #find trainable discriminator variable
        for var in self.gen_vars:
            print(var.name)

        if outputs ==2:
            self.y_ = tf.concat([tf.ones([tf.shape(self.y_conv)[0],1]), tf.zeros([tf.shape(self.y_conv)[0],1])], axis = 1, name= 'y_')
            self.fake_y_ =tf.concat([tf.zeros([tf.shape(self.fake_y_conv)[0],1]), tf.ones([tf.shape(self.fake_y_conv)[0],1])], axis = 1, name='fake_y_') #placeholder for fake representation
            self.gen_y_ =tf.concat([tf.ones([tf.shape(self.fake_y_conv)[0],1]), tf.zeros([tf.shape(self.fake_y_conv)[0],1])], axis = 1, name='gen_y_') #placeholder for fake representation
        else:
            self.y_ = tf.ones_like(self.y_conv, name= 'y_')
            self.fake_y_ =  tf.zeros_like(self.fake_y_conv, name= 'fake_y_')
            self.gen_y_ = tf.ones_like(self.fake_y_conv,name='gen_y_') #placeholder for fake representation

        if outputs ==2:
            self.d_cross_entropy = tf.add(tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.y_conv)) , tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.fake_y_, logits=self.fake_y_conv)))# reduce mean for discriminator
            self.d_cross_entropy_summary = tf.summary.scalar('d_loss',self.d_cross_entropy)
            self.gen_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.gen_y_, logits=self.fake_y_conv))#reduce mean for generator
            self.gen_cross_entropy_summary = tf.summary.scalar('g_loss',self.gen_cross_entropy)
            self.accuracy_real = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_conv,1), tf.argmax(self.y_,1)), tf.float32))#determine various accuracies
            self.accuracy_fake = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.fake_y_conv,1), tf.argmax(self.fake_y_,1)), tf.float32))
        else:
            self.d_cross_entropy = tf.add(tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv)) , tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=self.fake_y_, logits=self.fake_y_conv)))# reduce mean for discriminator
            self.d_cross_entropy_summary = tf.summary.scalar('d_loss',self.d_cross_entropy)
            self.gen_cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.gen_y_, logits=self.fake_y_conv))#reduce mean for generator
            self.gen_cross_entropy_summary = tf.summary.scalar('g_loss',self.gen_cross_entropy)
            self.accuracy_real = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(self.y_conv)), self.y_), tf.float32))#determine various accuracies
            self.accuracy_fake = tf.reduce_mean(tf.cast(tf.equal(tf.round(tf.sigmoid(self.fake_y_conv)), self.fake_y_), tf.float32))
        # self.accuracy_real = tf.reduce_mean(tf.cast(tf.equal(self.y_conv, self.y_), tf.float32))#determine various accuracies
        # self.accuracy_fake = tf.reduce_mean(tf.cast(tf.equal(self.fake_y_conv, self.fake_y_), tf.float32))
        self.accuracy_summary_real = tf.summary.scalar('accuracy_real',self.accuracy_real)
        self.accuracy_summary_fake = tf.summary.scalar('accuracy_fake',self.accuracy_fake)
        if outputs == 2:
            self.y_conv_summary = tf.summary.scalar('y_conv_right',tf.reduce_mean(self.y_conv[:,0]))
            self.fake_y_conv_summary = tf.summary.scalar('fake_y_conv_right',tf.reduce_mean(self.fake_y_conv[:,1]))
            self.y_conv_summary_w = tf.summary.scalar('y_conv_wrong',tf.reduce_mean(self.y_conv[:,1]))
            self.fake_y_conv_summary_w = tf.summary.scalar('fake_y_conv_wrong',tf.reduce_mean(self.fake_y_conv[:,0]))
            self.real_summary = tf.summary.merge([self.accuracy_summary_real,self.y_conv_summary,self.y_conv_summary_w,self.d_cross_entropy_summary])
            self.fake_summary = tf.summary.merge([self.accuracy_summary_fake,self.fake_y_conv_summary,self.fake_y_conv_summary_w ,self.gen_cross_entropy_summary])
        else:
            self.y_conv_summary = tf.summary.scalar('y_conv_right',tf.reduce_mean(self.y_conv))
            self.fake_y_conv_summary = tf.summary.scalar('fake_y_conv_right',tf.reduce_mean(self.fake_y_conv))
            self.real_summary = tf.summary.merge([self.accuracy_summary_real,self.y_conv_summary,self.d_cross_entropy_summary])
            self.fake_summary = tf.summary.merge([self.accuracy_summary_fake,self.fake_y_conv_summary,self.gen_cross_entropy_summary])

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.train_step = tf.train.AdamOptimizer(self.learning_rate,beta1=.5).minimize(self.d_cross_entropy, var_list=self.d_vars)
            self.gen_train_step = tf.train.AdamOptimizer(self.learning_rate,beta1=.5).minimize(self.gen_cross_entropy,var_list = self.gen_vars)

    def defineModels(self):
        # convolutions = [self.inputSize[2]] + self.convolutions
        self.y_conv = self.createDiscriminator(self.x,self.classes,self.inputSize,self.convolutions,self.fullyconnected,self.output) #real image discrimator
        # convolutions = [self.inputSize[2]] + self.convolutions
        self.fake_input = self.createGenerator(Z = self.Z, classes = self.classes,inputSize = self.Zsize, convolutions= self.convolutions[::-1],fullyconnected = None, output = self.inputSize) #generator
        self.fake_y_conv = self.createDiscriminator(self.fake_input,self.classes,self.inputSize,self.convolutions,self.fullyconnected,self.output, reuse = True)#fake image discrimator
        self.fake_input_summary = tf.summary.image("fake_inputs", tf.reshape(self.fake_input, [-1,inputSize[0],inputSize[1],inputSize[2]]),max_outputs = outputsFake)#show fake image
        self.real_input_summary = tf.summary.image("real_inputs", tf.reshape(self.x, [-1,inputSize[0],inputSize[1],inputSize[2]]),max_outputs = outputsReal)#show real image

    def defineVariables(self,inputSize,convolutions,fullyconnected,output,restore = False,fileName =None, classes = None):
        '''Make the instantiator, first make a lot of constants available'''
        self.inputSize = inputSize
        self.convolutions= [inputSize[2]] + convolutions
        self.restore = restore
        self.fullyconnected = fullyconnected
        self.output = output
        self.Zsize =100
        self.numClasses =classes
        self.fileName = fileName
        self.sess = tf.Session()#start the session
        self.my_gen = None
        '''Create the placeholders, then the discriminator and Generator'''
        self.Z = tf.placeholder(tf.float32, shape=[None, self.Zsize], name='Z') #random input
        self.classes = tf.placeholder(tf.float32, shape=[None, self.numClasses], name='class_inputs') #correct class
        self.x = tf.placeholder(tf.float32, shape=[None, inputSize[0]*inputSize[1]*inputSize[2]], name = 'true_input') #input image
        self.learning_rate = tf.placeholder(tf.float32)
        self.keep_prob = tf.placeholder(tf.float32)

    def generateZMaps(self):
        self.fake_z = self.createZMap(self.fake_input,self.classes,self.inputSize,self.convolutions,self.fullyconnected,self.output) #real image discrimator
        self.real_z = self.createZMap(self.x,self.classes,self.inputSize,self.convolutions,self.fullyconnected,self.output, reuse = True)#fake image discrimator
        self.x_reconfigured = self.createGenerator(Z = self.real_z, classes = self.classes,inputSize = self.Zsize, convolutions= self.convolutions[::-1],fullyconnected = None, output = self.inputSize, reuse = True)
        self.reconfig_inputs_summary = tf.summary.image("reconfig_inputs", tf.reshape(self.x_reconfigured, [-1,inputSize[0],inputSize[1],inputSize[2]]),max_outputs = reconfig_outputs)#show fake image
	self.input_summary_merge = tf.summary.merge([self.reconfig_inputs_summary, self.real_input_summary])
        t_vars = tf.trainable_variables()
        self.z_vars = [var for var in t_vars if 'z_' in var.name] #find trainable discriminator variable
        for var in self.z_vars:
            print(var.name)
        self.z_cross_entropy = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.Z, predictions=self.fake_z))#reduce mean for generator
        self.z_train_step = tf.train.AdamOptimizer(z_learning_rate,beta1=.5).minimize(self.z_cross_entropy, var_list=self.z_vars)
        self.z_cross_entropy_summary = tf.summary.scalar('z_loss',self.z_cross_entropy)

    def createZMap(self,x,classes,inputSize,convolutions,fullyconnected,output, reuse = False):
        flat3 = self.createDiscriminator(x,classes,inputSize,convolutions,fullyconnected,output,reuse = reuse, returnEarly =True)
        with tf.variable_scope("z_discriminator") as scope:
            if reuse: #get previous variable if we are reusing the discriminator but with fake images
                scope.reuse_variables()
            hid3 = DenseLayer(flat3, fullyconnected, act = tf.nn.relu, name = 'z_fcl')
            # self.keep_prob = tf.placeholder(tf.float32)
            # drop = InputLayer(tf.nn.dropout(hid3.outputs, self.keep_prob),name="Extra fucking dropout")
            # concat2 = ConcatLayer([drop, inputClass], 1, name ='d_concat_layer_2')
            y_conv = DenseLayer(hid3, self.Zsize, name = 'z_hidden_encode')
        return tf.nn.tanh(y_conv.outputs, name = 'z_tanh')

    def createGenerator(self,Z, classes, inputSize,convolutions,fullyconnected,output, reuse = False):
        '''Function to create the enerator and give its output
        Note that in convolutions, negative convolutions mean downsampling'''
        # Generator Net
        with tf.variable_scope("gen_generator") as scope:
            if reuse:
                scope.reuse_variables()
            inputs = InputLayer(Z, name='gen_inputs')
            inputClass =InputLayer(classes, name='gen_class_inputs_z')

            if self.numClasses is not None: inputs = ConcatLayer([inputs, inputClass], 1, name ='gen_concat_layer_z')
            numPools = sum([1 for i in convolutions if i < 0]) #count number of total convolutions
            print(numPools)
            xs, ys = output[0]/(2**numPools), output[1]/(2**numPools) #calculate start image size from numPools
            sizeDeconv = xs * ys * abs(convolutions[0])

            if fullyconnected is not None:
                inputs = DenseLayer(inputs, fullyconnected, act = tf.nn.leaky_relu, name = 'gen_fc')

            deconveInputFlat = DenseLayer(inputs, sizeDeconv, act = tf.nn.leaky_relu, name = 'gen_fdeconv')#dense layer to input to be reshaped
            deconveInput = ReshapeLayer(deconveInputFlat, (-1, xs, ys, abs(convolutions[0])), name = 'gen_unflatten')

            # convolutions.append(self.inputSize[2])
            convVals = [deconveInput]

            for i,v in enumerate(convolutions):#for every convolution
                if i < len(convolutions)-1:
                    '''The purpose of tile is to add outputs of correct image size
                    that are all 1s to represent a certain class'''
                    if self.numClasses is not None and i < len(convolutions)-1:#decide when to add the tiles
                        class_image = tf.tile(tf.expand_dims(tf.expand_dims(classes,1),1), (1,xs,ys,1))
                        class_image = InputLayer(class_image, name='gen_class_inputs_%i'%(i))
                        convVals[-1] = ConcatLayer([convVals[-1], class_image], 3, name ='gen_deconv_plus_classes_%i'%(i))

                    if convolutions[i] < 0:#decide whether to upsample
                        # convolutions[i] *= -1
                        xs *= 2
                        ys *= 2
                        stride  = (2,2)
                    else:
                        stride = (1,1)

                    if i == len(convolutions)-2:#if on the last step, use tanh instead of leaky_relu
                        convVals.append(DeConv2d(convVals[-1],abs(convolutions[i+1]), (5, 5), (xs,ys), stride,name='gen_fake_input',act =tf.nn.tanh))
                    else:
                        convVals.append(BatchNormLayer(DeConv2d(convVals[-1],abs(convolutions[i+1]), (5, 5), (xs,ys), stride,name='gen_deconv%i'%(i)),act =tf.nn.leaky_relu,is_train=True,name='gen_batch_norm%i'%(i)))
            return FlattenLayer(convVals[-1]).outputs #return flattened outputs


    def createDiscriminator(self,x,classes,inputSize,convolutions,fullyconnected,output,reuse=False, returnEarly = False):

        '''Create a discrimator, not the convolutions may be negative to represent
        downsampling'''
        prefix = 'd_'
        if returnEarly: prefix = "z_"
        with tf.variable_scope(prefix + "discriminator") as scope:
            if reuse: #get previous variable if we are reusing the discriminator but with fake images
                scope.reuse_variables()
            flatSize = inputSize[0]*inputSize[1]*inputSize[2]
            x_image = tf.reshape(x, [-1,inputSize[0],inputSize[1],inputSize[2]])#reshape image into right size
            xs, ys = inputSize[0],inputSize[1]
            numPools = sum([1 for i in convolutions if i < 0])
            print(numPools)
            inputs = InputLayer(x_image, name=prefix + 'disc_inputs')
            convVals = [inputs]#list of filters
            for i,v in enumerate(convolutions):
                '''Similarly tile for constant reference to class'''

                if i < len(convolutions)-1:#if it is negative, that means we pool on this step
                    if self.numClasses is not None:
                        class_image = tf.tile(tf.expand_dims(tf.expand_dims(classes,1),1), (1,xs,ys,1))
                        class_image = InputLayer(class_image, name=prefix + 'class_inputs_%i'%(i))
                        convVals[-1] = ConcatLayer([convVals[-1], class_image], 3, name =prefix + 'conv_plus_classes_%i'%(i))

                    pool=False
                    if convolutions[i+1] < 0:
                        # convolutions[i+1] *= -1
                        pool = True
                        xs, ys = xs/2, ys/2
                    #add necessary convolutional layers
                    if pool:
                        convVals.append(BatchNormLayer(Conv2d(convVals[-1], abs(convolutions[i+1]), (5, 5),strides = (2,2), name=prefix + 'conv1_%s'%(i)), act=tf.nn.leaky_relu,is_train=True ,name=prefix + 'batch_norm%s'%(i)))
                    else:
                        convVals.append(BatchNormLayer(Conv2d(convVals[-1], abs(convolutions[i+1]), (5, 5),strides = (1,1), name=prefix + 'conv1_%s'%(i)), act=tf.nn.leaky_relu,is_train=True ,name=prefix + 'batch_norm%s'%(i)))
                else:

                    # fully connecter layer
                    l,w,d = inputSize[0]/(2**numPools),inputSize[1]/(2**numPools),abs(convolutions[-1])
                    flat3 = FlattenLayer(convVals[-1], name = prefix + 'flatten')
                    if returnEarly: return flat3
                    if self.numClasses is not None:
                        inputClass =InputLayer(classes, name=prefix + 'class_inputs')
                        flat3 = ConcatLayer([flat3, inputClass], 1, name =prefix + 'concat_layer')
                    # hid3 = DenseLayer(concat, fullyconnected, act = tf.nn.leaky_relu, name = 'd_fcl')
                    # # self.keep_prob = tf.placeholder(tf.float32)
                    # drop = InputLayer(tf.nn.dropout(hid3.outputs, self.keep_prob),name="Extra fucking dropout")
                    # concat2 = ConcatLayer([drop, inputClass], 1, name ='d_concat_layer_2')
                    y_conv = DenseLayer(flat3, output, name = prefix + 'hidden_encode')
            return y_conv.outputs

    def train_z(self, iterations, batchLen = 50):
        for i in range(iterations):
            batch = self.getbatch(batchLen)
            z = np.random.uniform(-1, 1, size = [batchLen,self.Zsize])# define random z
            feed_dict = {self.Z:z,self.x: batch}
            _, z_cross_entropy_summary, input_summary_merge = self.sess.run([self.z_train_step, self.z_cross_entropy_summary,self.input_summary_merge],
            feed_dict=feed_dict)#train generator)
            self.train_writer.add_summary(z_cross_entropy_summary, i)
            if i % 5 == 0:
		self.train_writer.add_summary(input_summary_merge, i)
            if i % 30 == 0: print(i)

            if i % whenSave ==0:
                '''Save session'''
                if self.fileName is not None:
                    self.saver.save(self.sess, self.fileName)




    def train(self,iterations,batchLen = 50):
        '''Train the model'''
        print('Start Training')
        print('Loaded trainging')
        if get_video:
            self.imageBuffer = []
            self.imageFilePath = imageFilePath
            z_constant = np.random.uniform(-1, 1, size = [1,self.Zsize])
        for i in range(iterations):
            batch = self.getbatch(batchLen)# get batch
            # print('batch')
            # batch = batch[0],batch[1],newBatch
            # print(batch[0].shape)
            z = np.random.uniform(-1, 1, size = [batchLen,self.Zsize])# define random z

            if i % whenSave ==0:
                '''Save session'''
                if self.fileName is not None:
                    self.saver.save(self.sess, self.fileName)
            if self.numClasses is not None:
                feed_dict = {self.x: batch[0],self.Z: z, self.classes: batch[1], self.learning_rate: learning_rate}
            else:
                feed_dict = {self.x: batch,self.Z: z, self.learning_rate: learning_rate}

            _, real_input_summary, real_summary= self.sess.run([self.train_step,self.real_input_summary,self.real_summary],
            feed_dict=feed_dict) #train discrimator

            if self.numClasses is not None:
                feed_dict = {self.Z: z, self.classes: batch[1], self.learning_rate: learning_rate}
            else:
                feed_dict = {self.Z: z, self.learning_rate: learning_rate}

            _, fake_input_summary,fake_summary = self.sess.run([self.gen_train_step,self.fake_input_summary,self.fake_summary],
            feed_dict=feed_dict)#train generator
            if get_video and i % whenAddPicture == 0:
                feed_dict.update({self.Z: z_constant})
                fake_input = self.sess.run([self.fake_input],
                feed_dict=feed_dict)
                self.imageBuffer.append(np.reshape(fake_input[0],[self.inputSize[0],self.inputSize[1],self.inputSize[2]]))
            # gen_train, fake_input_summary,gen_cross_entropy_summary= self.sess.run([self.gen_train_step,self.fake_input_summary,self.gen_cross_entropy_summary],
            # feed_dict=feed_dict)#train generator
            if i % tbWhenSavePicture ==0:#push images every 50 iterations
                self.train_writer.add_summary(fake_input_summary, i)
                self.train_writer.add_summary(real_input_summary, i)

            if i % whenSaveMovie == 0:
                if get_video:
                    saveMovie.writeMovie(self.imageFilePath,self.imageBuffer)
                    # self.imageBuffer = []
            self.train_writer.add_summary(fake_summary, i)
            self.train_writer.add_summary(real_summary, i)

    def getbatch(self,batchLen):
        if self.my_gen is None:
            self.my_gen = importImg.get_batches(batchLen,folder, self.inputSize[0],self.inputSize[1], box=box)
        x_batch = next(self.my_gen,None)
        while x_batch is None:#when the generator is done, instantiate a new one
            self.my_gen = importImg.get_batches(batchLen,folder, self.inputSize[0],self.inputSize[1], box=box)
            print("Ran Out!")
            x_batch = next(self.my_gen,None)
        print(x_batch.shape)
        return np.reshape(x_batch, [-1, self.inputSize[0]*self.inputSize[1]*self.inputSize[2]])

    def getbatchGenerator(self,batchLen):
        data,classes,one_hot_classes  = importCIFAR.load_training_data()

        for i in range(0,len(classes),batchLen):
            j = min(i+batchLen,len(classes))
            c_data = data[i:j].reshape([batchLen, self.inputSize[0]*self.inputSize[1]*self.inputSize[2]])
            yield c_data, one_hot_classes[i:j]


    def iterateOverVariables(self,numPictures):
        print('Loaded trainging')
        examples = 9
        length = 3
        height = 3
        if get_video:
            # self.imageBuffer = [[] for i in range(self.Zsize/sizeDiff)]
            self.imageBuffer2 =[]
            self.imageFilePath = imageFilePath
            z = np.zeros([examples,self.Zsize],dtype=np.float32)
        with imageio.get_writer(imageFilePath, mode='I') as writer:

            for i in range(-numPictures,numPictures):
                print(i)
                for j in range(examples):
                    # print(float(i)/numPictures)
                    z[j][j] = float(i)/numPictures
                feed_dict = {self.Z: z}
                fake_input = self.sess.run([self.fake_input],
                feed_dict=feed_dict)
                mega = np.reshape(fake_input[0],[-1,self.inputSize[0],self.inputSize[1],self.inputSize[2]])

                mega = [np.concatenate(mega[j:j+length],axis = 1) for j in range(0,examples,length)]
                mega = np.concatenate(mega[0:height],axis =0)
                # print(fake_input[0][0].shape)
                # print(len(self.imageBuffer[0]))
                # print(np.reshape(fake_input[0][0],[self.inputSize[0],self.inputSize[1],self.inputSize[2]]).shape)
                # [self.imageBuffer[j].append(np.reshape(fake_input[0][j],[self.inputSize[0],self.inputSize[1],self.inputSize[2]])) for j in range(self.Zsize/sizeDiff)]
                # self.imageBuffer2.append(mega)
                writer.append_data(mega)

            for i in range(numPictures,-numPictures,-1):
                print(i)
                for j in range(examples):
                    z[j][j] = float(i)/numPictures
                feed_dict = {self.Z: z}
                fake_input = self.sess.run([self.fake_input],
                feed_dict=feed_dict)
                mega = np.reshape(fake_input[0],[-1,self.inputSize[0],self.inputSize[1],self.inputSize[2]])

                mega = [np.concatenate(mega[j:j+length],axis = 1) for j in range(0,examples,length)]
                mega = np.concatenate(mega[0:height],axis =0)
                # print(fake_input[0][0].shape)
                # print(len(self.imageBuffer[0]))
                # print(np.reshape(fake_input[0][0],[self.inputSize[0],self.inputSize[1],self.inputSize[2]]).shape)
                # [self.imageBuffer[j].append(np.reshape(fake_input[0][j],[self.inputSize[0],self.inputSize[1],self.inputSize[2]])) for j in range(self.Zsize/sizeDiff)]
                # self.imageBuffer2.append(mega)
                writer.append_data(mega)
                    # self.imageBuffer = []
        # for j in range(self.Zsize/sizeDiff):
            # print(self.imageBuffer[j])
            # print(len(self.imageBuffer[j]))
            # saveMovie.writeMovie(self.imageFilePath+str(j)+".gif",self.imageBuffer[j])
        # saveMovie.writeMovie(self.imageFilePath,self.imageBuffer2)


if __name__ == "__main__":

    myDiscriminator = GAN(inputSize = inputSize,convolutions = convolutions, fullyconnected = fullyconnected, output = 1, fileName = model_filepath,restore = restore, classes = classes)
    myDiscriminator.train(iterations, batch_size)
    #myDiscriminator.iterateOverVariables(40)

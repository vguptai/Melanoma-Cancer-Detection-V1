import tensorflow as tf
from config import *

tf.set_random_seed(tensorflowSeed)

class convNetModel:

    optimizer = None
    accuracy = None
    cost = None
    prediction = None
    correct = None
    x = None
    y = None
    keep_rate = None
    learning_rate = None

    def __init__(self,decaySteps):
        self.x = tf.placeholder('float', [None, imageSizeX,imageSizeY,numChannels])
        self.y = tf.placeholder('float')
        self.keep_rate = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        self._setupNetwork(decaySteps)

    def _conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _maxpool2d(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def _buildNetworkOld(self):
        weights = {'W_conv1': tf.Variable(tf.random_normal([5,5, numChannels, 32],seed=opsSeed)),
                   'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64],seed=opsSeed)),
                   'W_conv3': tf.Variable(tf.random_normal([5, 5, 64, 64],seed=opsSeed)),
                   'W_conv4': tf.Variable(tf.random_normal([3, 3, 64, 64],seed=opsSeed)),
                   'W_conv5': tf.Variable(tf.random_normal([3, 3, 64, 64],seed=opsSeed)),
                   'W_conv6': tf.Variable(tf.random_normal([3, 3, 64, 64],seed=opsSeed)),
                   'W_fc_1': tf.Variable(tf.random_normal([imageSizeX/8 * imageSizeY/8 * 64, 1024],seed=opsSeed)),
                   'W_fc_2': tf.Variable(tf.random_normal([1024, 512],seed=opsSeed)),
                   'out': tf.Variable(tf.random_normal([512, n_classes],seed=opsSeed))}

        biases = {'b_conv1': tf.Variable(tf.random_normal([32],seed=opsSeed)),
                  'b_conv2': tf.Variable(tf.random_normal([64],seed=opsSeed)),
                  'b_conv3': tf.Variable(tf.random_normal([64],seed=opsSeed)),
                  'b_conv4': tf.Variable(tf.random_normal([64],seed=opsSeed)),
                  'b_conv5': tf.Variable(tf.random_normal([64],seed=opsSeed)),
                  'b_conv6': tf.Variable(tf.random_normal([64],seed=opsSeed)),
                  'b_fc_1': tf.Variable(tf.random_normal([1024],seed=opsSeed)),
                  'b_fc_2': tf.Variable(tf.random_normal([512],seed=opsSeed)),
                  'out': tf.Variable(tf.random_normal([n_classes],seed=opsSeed))}


        conv1 = tf.nn.relu(self._conv2d(self.x, weights['W_conv1']) + biases['b_conv1'])
        if enableLocalResponseNormalization:
            conv1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1_1')

        conv2 = tf.nn.relu(self._conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        if enableLocalResponseNormalization:
            conv2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='norm1_1')

        conv2 = self._maxpool2d(conv2)

        conv3 = tf.nn.relu(self._conv2d(conv2,weights['W_conv3'])+ biases['b_conv3'])
        conv4 = tf.nn.relu(self._conv2d(conv3,weights['W_conv4'])+ biases['b_conv4'])
        conv4 = self._maxpool2d(conv4)

        conv5 = tf.nn.relu(self._conv2d(conv4,weights['W_conv5'])+ biases['b_conv5'])
        conv6 = tf.nn.relu(self._conv2d(conv5,weights['W_conv6'])+ biases['b_conv6'])
        conv6 = self._maxpool2d(conv6)

        with tf.name_scope('fc_1') as scope:
            fc_1 = tf.reshape(conv6, [-1, imageSizeX/8 * imageSizeY/8 * 64])
            fc_1 = tf.matmul(fc_1, weights['W_fc_1']) + biases['b_fc_1']
            if enableBatchNormalization:
                fc_1 = self._add_contrib_batch_norm_layer(scope,fc_1,self.is_training)
            fc_1 = tf.nn.relu(fc_1)
            fc_1 = tf.nn.dropout(fc_1, self.keep_rate, seed=dropoutSeed)

        with tf.name_scope('fc_2') as scope:
            fc_2 = tf.matmul(fc_1, weights['W_fc_2']) + biases['b_fc_2']
            if enableBatchNormalization:
                fc_2 = self._add_contrib_batch_norm_layer(scope,fc_2,self.is_training)
            fc_2 = tf.nn.relu(fc_2)
            fc_2 = tf.nn.dropout(fc_2, self.keep_rate, seed=dropoutSeed)

        output = tf.matmul(fc_2, weights['out']) + biases['out']
        return output

    def _buildNetwork(self):
        weights = {'W_fc_1': tf.Variable(tf.random_normal([imageSizeX/8 * imageSizeY/8 * 64, 1024],seed=opsSeed)),
                   'W_fc_2': tf.Variable(tf.random_normal([1024, 512],seed=opsSeed)),
                   'out': tf.Variable(tf.random_normal([512, n_classes],seed=opsSeed))}

        biases = {'b_fc_1': tf.Variable(tf.random_normal([1024],seed=opsSeed)),
                  'b_fc_2': tf.Variable(tf.random_normal([512],seed=opsSeed)),
                  'out': tf.Variable(tf.random_normal([n_classes],seed=opsSeed))}

        conv1 = self.addConvLayer(self.x,3,3,numChannels,32,enableLocalResponseNormalization,enableBatchNormalization,'conv1',self.is_training)
        conv2 = self.addConvLayer(conv1,3,3,32,64,enableLocalResponseNormalization,enableBatchNormalization,'conv2',self.is_training)
        conv2 = self._maxpool2d(conv2)

        conv3 = self.addConvLayer(conv2,3,3,64,64,enableLocalResponseNormalization,enableBatchNormalization,'conv3',self.is_training)
        conv4 = self.addConvLayer(conv3,3,3,64,64,enableLocalResponseNormalization,enableBatchNormalization,'conv4',self.is_training)
        conv4 = self._maxpool2d(conv4)

        conv5 = self.addConvLayer(conv4,3,3,64,64,enableLocalResponseNormalization,enableBatchNormalization,'conv5',self.is_training)
        conv6 = self.addConvLayer(conv5,3,3,64,64,enableLocalResponseNormalization,enableBatchNormalization,'conv6',self.is_training)
        conv6 = self._maxpool2d(conv6)

        with tf.name_scope('fc_1') as scope:
            fc_1 = tf.reshape(conv6, [-1, imageSizeX/8 * imageSizeY/8 * 64])
            fc_1 = tf.matmul(fc_1, weights['W_fc_1']) + biases['b_fc_1']
            if enableBatchNormalization:
                fc_1 = self._add_contrib_batch_norm_layer(scope,fc_1,self.is_training)
            fc_1 = tf.nn.relu(fc_1)
            fc_1 = tf.nn.dropout(fc_1, self.keep_rate, seed=dropoutSeed)

        with tf.name_scope('fc_2') as scope:
            fc_2 = tf.matmul(fc_1, weights['W_fc_2']) + biases['b_fc_2']
            if enableBatchNormalization:
                fc_2 = self._add_contrib_batch_norm_layer(scope,fc_2,self.is_training)
            fc_2 = tf.nn.relu(fc_2)
            fc_2 = tf.nn.dropout(fc_2, self.keep_rate, seed=dropoutSeed)

        output = tf.matmul(fc_2, weights['out']) + biases['out']
        return output


    def addConvLayer(self,input,filterSizeX,filterSizeY,numInputChannels,numOutputChannels,enableLocalResponseNormalization,enableBatchNormalization,layerName,is_training):
        with tf.name_scope(layerName) as scope:
            with tf.name_scope("weights"):
                weights = tf.Variable(tf.random_normal([filterSizeX,filterSizeY, numInputChannels,numOutputChannels],seed=opsSeed))
            with tf.name_scope("biases"):
                biases = tf.Variable(tf.random_normal([numOutputChannels],seed=opsSeed))
            conv = self._conv2d(input, weights) + biases
            if enableLocalResponseNormalization:
                conv = tf.nn.lrn(conv, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='lrn')
            if enableBatchNormalization:
                with tf.name_scope("batch_norm") as scope:
                    conv = self._add_contrib_batch_norm_layer(scope,conv,is_training)
            conv = tf.nn.relu(conv)
            return conv

    def _add_contrib_batch_norm_layer(self,scope,x,is_training,decay=0.99):
        return tf.contrib.layers.batch_norm(x,decay=decay, is_training=is_training,updates_collections=None,scope=scope,reuse=True,center=True)


    def _buildNetworkOld(self):
	weights = {'W_conv1': tf.Variable(tf.random_normal([5,5, numChannels, 32],seed=opsSeed)),
                   'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64],seed=opsSeed)),
                   'W_conv3': tf.Variable(tf.random_normal([5, 5, 64, 64],seed=opsSeed)),
                   'W_conv4': tf.Variable(tf.random_normal([3, 3, 64, 64],seed=opsSeed)),
                   'W_conv5': tf.Variable(tf.random_normal([3, 3, 64, 64],seed=opsSeed)),
                   'W_fc_1': tf.Variable(tf.random_normal([imageSizeX/32 * imageSizeY/32 * 64, 1024],seed=opsSeed)),
                   'W_fc_2': tf.Variable(tf.random_normal([1024, 512],seed=opsSeed)),
                   'out': tf.Variable(tf.random_normal([1024, n_classes],seed=opsSeed))}

	biases = {'b_conv1': tf.Variable(tf.random_normal([32],seed=opsSeed)),
                  'b_conv2': tf.Variable(tf.random_normal([64],seed=opsSeed)),
                  'b_conv3': tf.Variable(tf.random_normal([64],seed=opsSeed)),
                  'b_conv4': tf.Variable(tf.random_normal([64],seed=opsSeed)),
                  'b_conv5': tf.Variable(tf.random_normal([64],seed=opsSeed)),
                  'b_fc_1': tf.Variable(tf.random_normal([1024],seed=opsSeed)),
                  'b_fc_2': tf.Variable(tf.random_normal([512],seed=opsSeed)),
                  'out': tf.Variable(tf.random_normal([n_classes],seed=opsSeed))}


        conv1 = tf.nn.relu(self._conv2d(self.x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = self._maxpool2d(conv1)
        #conv1 = tf.nn.dropout(conv1,self.keep_rate, seed=dropoutSeed)

        conv2 = tf.nn.relu(self._conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = self._maxpool2d(conv2)
        #conv2 = tf.nn.dropout(conv2,self.keep_rate, seed=dropoutSeed)

        conv3 = tf.nn.relu(self._conv2d(conv2,weights['W_conv3'])+ biases['b_conv3'])
        conv3 = self._maxpool2d(conv3)
        #conv3 = tf.nn.dropout(conv3,self.keep_rate, seed=dropoutSeed)

        conv4 = tf.nn.relu(self._conv2d(conv3,weights['W_conv4'])+ biases['b_conv4'])
        conv4 = self._maxpool2d(conv4)

        conv5 = tf.nn.relu(self._conv2d(conv4,weights['W_conv5'])+ biases['b_conv5'])
        conv5 = self._maxpool2d(conv5)

        fc_1 = tf.reshape(conv5, [-1, imageSizeX/32 * imageSizeY/32 * 64])
        fc_1 = tf.nn.relu(tf.matmul(fc_1, weights['W_fc_1']) + biases['b_fc_1'])
        fc_1 = tf.nn.dropout(fc_1, self.keep_rate, seed=dropoutSeed)

        fc_2 = tf.nn.relu(tf.matmul(fc_1, weights['W_fc_2']) + biases['b_fc_2'])
        fc_2 = tf.nn.dropout(fc_2, self.keep_rate, seed=dropoutSeed)

        output = tf.matmul(fc_1, weights['out']) + biases['out']

        return output

    def _setupNetwork(self,decaySteps):

        self.prediction = self._buildNetwork()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.learning_rate = tf.train.exponential_decay(learningRateInitial,self.global_step,decaySteps,learningRateDecayFactor,staircase=True)
        #self.learning_rate = learningRateInitial
        #self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost,global_step=self.global_step)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost,global_step=self.global_step)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost,global_step=self.global_step)
        #self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate,0.9,momentum=0.9,epsilon=1).minimize(self.cost,global_step=self.global_step)
        self.correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))

    def getGlobalStep(self):
        return self.global_step

    def train(self,sess,trainX,trainY):
        return sess.run([self.optimizer, self.cost], feed_dict={self.x: trainX, self.y: trainY, self.keep_rate: training_keep_rate, self.is_training:True})

    def test(self,testX,testY):
        return self.accuracy.eval({self.x: testX, self.y: testY, self.keep_rate: testing_keep_rate, self.is_training:False})

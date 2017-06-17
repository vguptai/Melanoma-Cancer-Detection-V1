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
        self._setupNetwork(decaySteps)

    def _conv2d(self,x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def _maxpool2d(self,x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


    def _buildNetwork(self):
	weights = {'W_conv1': tf.Variable(tf.random_normal([5, 5, numChannels, 32],seed=opsSeed)),
                   'W_conv2': tf.Variable(tf.random_normal([5, 5, 32, 64],seed=opsSeed)),
                   'W_conv3': tf.Variable(tf.random_normal([5, 5, 64, 64],seed=opsSeed)),
                   'W_fc': tf.Variable(tf.random_normal([imageSizeX/8 * imageSizeY/8 * 64, 1024],seed=opsSeed)),
                   'out': tf.Variable(tf.random_normal([1024, n_classes],seed=opsSeed))}

	biases = {'b_conv1': tf.Variable(tf.random_normal([32],seed=opsSeed)),
                  'b_conv2': tf.Variable(tf.random_normal([64],seed=opsSeed)),
                  'b_conv3': tf.Variable(tf.random_normal([64],seed=opsSeed)),
                  'b_fc': tf.Variable(tf.random_normal([1024],seed=opsSeed)),
                  'out': tf.Variable(tf.random_normal([n_classes],seed=opsSeed))}


        conv1 = tf.nn.relu(self._conv2d(self.x, weights['W_conv1']) + biases['b_conv1'])
        conv1 = self._maxpool2d(conv1)

        conv2 = tf.nn.relu(self._conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
        conv2 = self._maxpool2d(conv2)

        conv3 = tf.nn.relu(self._conv2d(conv2,weights['W_conv3'])+ biases['b_conv3'])
        conv3 = self._maxpool2d(conv3)

        fc = tf.reshape(conv3, [-1, imageSizeX/8 * imageSizeY/8 * 64])
        fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])
        fc = tf.nn.dropout(fc, self.keep_rate, seed=dropoutSeed)

        output = tf.matmul(fc, weights['out']) + biases['out']

        return output

    def _setupNetwork(self,decaySteps):

        self.prediction = self._buildNetwork()
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.prediction, labels=self.y))
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        #self.learning_rate = tf.train.exponential_decay(learningRateInitial,self.global_step,decaySteps,learningRateDecayFactor,staircase=True)
        #self.optimizer = tf.train.AdamOptimizer().minimize(self.cost,global_step=self.global_step)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        #self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost,global_step=self.global_step)
        self.correct = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct, 'float'))

    def getGlobalStep(self):
        return self.global_step

    def train(self,sess,trainX,trainY):
        return sess.run([self.optimizer, self.cost], feed_dict={self.x: trainX, self.y: trainY, self.keep_rate: training_keep_rate})

    def test(self,testX,testY):
        return self.accuracy.eval({self.x: testX, self.y: testY, self.keep_rate: testing_keep_rate})

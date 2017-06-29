import tensorflow as tf
import numpy as np
from tensorflow.python.framework import tensor_shape

class InceptionV3:

	bottleneckTensor = None
	finalTensor = None
	groundTruthInput = None
	trainStep = None
	evaluationStep = None
	bottleneckInput = None
	inceptionGraph = None
	jpeg_data_tensor = None
	distortion_image_data_input_placeholder = None
	distort_image_data_operation = None
	keep_rate = 0.9
	learning_rate = None
	global_step = None
	is_training = None

	def __init__(self,modelPath):
		self._create_inception_graph(modelPath)

	def _create_inception_graph(self,modelPath):
		with tf.Graph().as_default() as self.inceptionGraph:
			with tf.gfile.FastGFile(modelPath, 'rb') as f:
					graph_def = tf.GraphDef()
					graph_def.ParseFromString(f.read())
					self.bottleneckTensor, self.jpeg_data_tensor, resized_input_tensor, self.decoded_jpeg_data_tensor = (tf.import_graph_def(graph_def, name='', return_elements=[BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME,RESIZED_INPUT_TENSOR_NAME,DECODED_JPEG_DATA_TENSOR_NAME]))


	def create_learning_rate(self,FLAGS,global_step,num_batches_per_epoch):
		if FLAGS.learning_rate_type == "const":
			print "Setting up a constant learning rate:"+str(FLAGS.learning_rate)
			self.learning_rate = FLAGS.learning_rate
		elif FLAGS.learning_rate_type == "exp_decay":
			decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
			print "Setting up an exponentially decaying learning rate:"+str(FLAGS.learning_rate)+":"+str(decay_steps)+":"+str(FLAGS.learning_rate_decay_factor)
			self.learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step,decay_steps,FLAGS.learning_rate_decay_factor,staircase=True)
		else:
			raise ValueError('Incorrect Learning Rate Type...')

	def _add_non_bn_fully_connected_layer(self,input_to_layer,input_size,output_size,layer_name,keep_rate):
		with tf.name_scope(layer_name):
			with tf.name_scope('weights'):
				initial_value_weights = tf.truncated_normal([input_size, output_size],stddev=0.001)
				layer_weights = tf.Variable(initial_value_weights, name='final_weights')
			with tf.name_scope('biases'):
				layer_biases = tf.Variable(tf.zeros([output_size]), name='final_biases')
			with tf.name_scope('Wx_plus_b'):
				logits_bn = tf.matmul(input_to_layer, layer_weights) + layer_biases
				logits_bn = tf.nn.relu(logits_bn)
				logits_bn = tf.nn.dropout(logits_bn, keep_rate)
		return logits_bn


	def _add_batch_norm(self,scope,x,is_training,reuse=None,epsilon=0.001,decay=0.99):
		with tf.variable_scope(scope,reuse=reuse):
			input_last_dimension = x.get_shape().as_list()[-1]
			#BN Hyperparams
			scale = tf.get_variable("scale", input_last_dimension, initializer=tf.constant_initializer(1.0), trainable=True)
			beta = tf.get_variable("beta", input_last_dimension, initializer=tf.constant_initializer(0.0), trainable=True)
			#Population Mean/Variance to be used while testing
			pop_mean = tf.get_variable("pop_mean",input_last_dimension, initializer=tf.constant_initializer(0.0), trainable=False)
			pop_var = tf.get_variable("pop_var", input_last_dimension, initializer=tf.constant_initializer(1.0), trainable=False)

			if is_training:
				#Mean and Variance of the logits
				batch_mean, batch_var = tf.nn.moments(x,range(len(x.get_shape().as_list())-1))
				train_mean = tf.assign(pop_mean,pop_mean * decay + batch_mean * (1 - decay))
				train_var = tf.assign(pop_var,pop_var * decay + batch_var * (1 - decay))
				with tf.control_dependencies([train_mean, train_var]):
					logits_bn = tf.nn.batch_normalization(x,batch_mean, batch_var, beta, scale, epsilon)
			else:
				logits_bn = tf.nn.batch_normalization(x,pop_mean, pop_var, beta, scale, epsilon)
			return logits_bn

	def _add_contrib_batch_norm_layer(self,scope,x,is_training,decay=0.99):
		return tf.contrib.layers.batch_norm(x,decay=decay, is_training=is_training,updates_collections=None,scope=scope,reuse=True,center=True)

	def _add_contrib_bn_fully_connected_layer(self,input_to_layer,input_size,output_size,layer_name,keep_rate,is_training):
		with tf.name_scope(layer_name):
			with tf.name_scope('weights'):
				initial_value_weights = tf.truncated_normal([input_size, output_size],stddev=0.001)
				layer_weights = tf.Variable(initial_value_weights, name='final_weights')
			with tf.name_scope('biases'):
				layer_biases = tf.Variable(tf.zeros([output_size]), name='final_biases')
			with tf.name_scope('Wx_plus_b'):
				#Calculate the logits
				logits = tf.matmul(input_to_layer, layer_weights)
			with tf.name_scope('batch_norm') as scope:
				#Batch Normalization
				logits_bn = self._add_contrib_batch_norm_layer(scope,logits,is_training)
			#Non Linearity
			logits_bn = tf.nn.relu(logits_bn)
			#Dropout
			logits_bn = tf.nn.dropout(logits_bn, keep_rate)
		return logits_bn

	# def _add_bn_fully_connected_layer(self,input_to_layer,input_size,output_size,layer_name,keep_rate,is_training):
	# 	with tf.name_scope(layer_name):
	# 		with tf.name_scope('weights'):
	# 			initial_value_weights = tf.truncated_normal([input_size, output_size],stddev=0.001)
	# 			layer_weights = tf.Variable(initial_value_weights, name='final_weights')
	# 		with tf.name_scope('biases'):
	# 			layer_biases = tf.Variable(tf.zeros([output_size]), name='final_biases')
	# 		with tf.name_scope('Wx_plus_b'):
	# 			#Calculate the logits
	# 			logits = tf.matmul(input_to_layer, layer_weights)
	# 		with tf.name_scope('batch_norm') as scope:
	# 			#Batch Normalization
	# 			logits_bn = tf.cond(is_training,
	# 			lambda: self._add_batch_norm(scope,logits,True,None),
    #     		lambda: self._add_batch_norm(scope,logits,False,True))
	# 		#Non Linearity
	# 		logits_bn = tf.nn.relu(logits_bn)
	# 		#Dropout
	# 		logits_bn = tf.nn.dropout(logits_bn, keep_rate)
	# 	return logits_bn

	def _add_fully_connected_layer(self,input_to_layer,input_size,output_size,layer_name,keep_rate,is_training, FLAGS):
		if FLAGS.use_batch_normalization:
			print "Batch normalization is turned on..."
			return self._add_contrib_bn_fully_connected_layer(input_to_layer,input_size,output_size,layer_name,keep_rate,is_training)
		else:
			print "Batch normalization is turned off..."
			return self._add_non_bn_fully_connected_layer(input_to_layer,input_size,output_size,layer_name,keep_rate)

	def add_final_training_ops(self,class_count, final_tensor_name, optimizer_name, num_batches_per_epoch, FLAGS):
		with self.inceptionGraph.as_default():
			with tf.name_scope('input'):
				self.bottleneckInput = tf.placeholder_with_default(self.bottleneckTensor, shape=[None, BOTTLENECK_TENSOR_SIZE],name='BottleneckInputPlaceholder')
				self.groundTruthInput = tf.placeholder(tf.float32,[None, class_count],name='GroundTruthInput')
				self.keep_rate = tf.placeholder(tf.float32, name='dropout_keep_rate')
				self.is_training_ph = tf.placeholder(tf.bool, name='is_training_ph')

			layer_name = 'final_minus_2_training_ops'
			logits_final_minus_2 = self._add_fully_connected_layer(self.bottleneckInput,BOTTLENECK_TENSOR_SIZE,FINAL_MINUS_2_LAYER_SIZE,layer_name,self.keep_rate,self.is_training_ph,FLAGS)

			layer_name = 'final_minus_1_training_ops'
			logits_final_minus_1 = self._add_fully_connected_layer(logits_final_minus_2,FINAL_MINUS_2_LAYER_SIZE,FINAL_MINUS_1_LAYER_SIZE,layer_name,self.keep_rate,self.is_training_ph,FLAGS)

			layer_name = 'final_training_ops'
			with tf.name_scope(layer_name):
			    with tf.name_scope('weights'):
			    	initial_value = tf.truncated_normal([FINAL_MINUS_1_LAYER_SIZE, class_count],stddev=0.001)
			      	layer_weights = tf.Variable(initial_value, name='final_weights')
			    with tf.name_scope('biases'):
			      	layer_biases = tf.Variable(tf.zeros([class_count]), name='final_biases')
			    with tf.name_scope('Wx_plus_b'):
					logits = tf.matmul(logits_final_minus_1, layer_weights) + layer_biases

			self.finalTensor = tf.nn.softmax(logits, name=final_tensor_name)
			with tf.name_scope('cross_entropy'):
				self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.groundTruthInput, logits=logits)
			    	with tf.name_scope('total'):
			    		self.cross_entropy_mean = tf.reduce_mean(self.cross_entropy)

			self.global_step = tf.Variable(0, name='global_step', trainable=False)
			self.create_learning_rate(FLAGS,self.global_step,num_batches_per_epoch)

			with tf.name_scope('train'):
				if optimizer_name == "sgd":
					optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
				elif optimizer_name == "adam":
					optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
				elif optimizer_name == "rmsprop":
					optimizer = tf.train.RMSPropOptimizer(self.learning_rate,FLAGS.rmsprop_decay,momentum=FLAGS.rmsprop_momentum,epsilon=FLAGS.rmsprop_epsilon)
				else:
					raise ValueError('Incorrect Optimizer Type...')
				self.trainStep = optimizer.minimize(self.cross_entropy_mean,global_step=self.global_step)

	def add_evaluation_step(self):
		with self.inceptionGraph.as_default():
			with tf.name_scope('accuracy'):
				with tf.name_scope('correct_prediction'):
					prediction = tf.argmax(self.finalTensor, 1)
					correctPrediction = tf.equal(prediction, tf.argmax(self.groundTruthInput, 1))
			with tf.name_scope('accuracy'):
				self.evaluationStep = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))
			return self.evaluationStep, prediction

	def train_step(self,sess,train_bottlenecks,train_ground_truth,dropout_keep_rate):
		#print self.global_step.eval()
		return sess.run([self.trainStep,self.cross_entropy_mean],feed_dict={self.bottleneckInput: train_bottlenecks,self.groundTruthInput: train_ground_truth, self.keep_rate:dropout_keep_rate, self.is_training_ph:True})

	def evaluate(self,sess,data_bottlenecks,data_ground_truth):
		accuracy, crossEntropyValue = sess.run([self.evaluationStep, self.cross_entropy_mean],feed_dict={self.bottleneckInput: data_bottlenecks,self.groundTruthInput: data_ground_truth, self.keep_rate:1, self.is_training_ph:False})
		return accuracy,crossEntropyValue

	def run_bottleneck_on_image(self,sess, image_data):
	  #bottleneck_values = sess.run(self.bottleneckTensor,{self.jpeg_data_tensor: image_data})
	  bottleneck_values = sess.run(self.bottleneckTensor,{self.decoded_jpeg_data_tensor: image_data})
	  bottleneck_values = np.squeeze(bottleneck_values)
	  return bottleneck_values

	def distort_image(self,sess,image_data):
		return sess.run(self.distort_image_data_operation ,{self.distortion_image_data_input_placeholder: image_data})

	def add_input_distortions(self, flip_left_right, random_crop, random_scale,
	                          random_brightness):
	  """Creates the operations to apply the specified distortions.
	  During training it can help to improve the results if we run the images
	  through simple distortions like crops, scales, and flips. These reflect the
	  kind of variations we expect in the real world, and so can help train the
	  model to cope with natural data more effectively. Here we take the supplied
	  parameters and construct a network of operations to apply them to an image.
	  Cropping
	  ~~~~~~~~
	  Cropping is done by placing a bounding box at a random position in the full
	  image. The cropping parameter controls the size of that box relative to the
	  input image. If it's zero, then the box is the same size as the input and no
	  cropping is performed. If the value is 50%, then the crop box will be half the
	  width and height of the input. In a diagram it looks like this:
	  <       width         >
	  +---------------------+
	  |                     |
	  |   width - crop%     |
	  |    <      >         |
	  |    +------+         |
	  |    |      |         |
	  |    |      |         |
	  |    |      |         |
	  |    +------+         |
	  |                     |
	  |                     |
	  +---------------------+
	  Scaling
	  ~~~~~~~
	  Scaling is a lot like cropping, except that the bounding box is always
	  centered and its size varies randomly within the given range. For example if
	  the scale percentage is zero, then the bounding box is the same size as the
	  input and no scaling is applied. If it's 50%, then the bounding box will be in
	  a random range between half the width and height and full size.
	  Args:
	    flip_left_right: Boolean whether to randomly mirror images horizontally.
	    random_crop: Integer percentage setting the total margin used around the
	    crop box.
	    random_scale: Integer percentage of how much to vary the scale by.
	    random_brightness: Integer range to randomly multiply the pixel values by.
	    graph.
	  Returns:
	    The jpeg input layer and the distorted result tensor.
	  """
	  print "Setting up image distortion operations..."
	  #jpeg_data = tf.placeholder(tf.string, name='DistortJPGInput')
	  #decoded_image = tf.image.decode_jpeg(jpeg_data, channels=MODEL_INPUT_DEPTH)
	  with self.inceptionGraph.as_default():
		  decoded_image_as_float = tf.placeholder('float', [None,None,MODEL_INPUT_DEPTH])
		  decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
		  margin_scale = 1.0 + (random_crop / 100.0)
		  resize_scale = 1.0 + (random_scale / 100.0)
		  margin_scale_value = tf.constant(margin_scale)
		  resize_scale_value = tf.random_uniform(tensor_shape.scalar(),
		                                         minval=1.0,
		                                         maxval=resize_scale)
		  scale_value = tf.multiply(margin_scale_value, resize_scale_value)
		  precrop_width = tf.multiply(scale_value, MODEL_INPUT_WIDTH)
		  precrop_height = tf.multiply(scale_value, MODEL_INPUT_HEIGHT)
		  precrop_shape = tf.stack([precrop_height, precrop_width])
		  precrop_shape_as_int = tf.cast(precrop_shape, dtype=tf.int32)
		  precropped_image = tf.image.resize_bilinear(decoded_image_4d,
		                                              precrop_shape_as_int)
		  precropped_image_3d = tf.squeeze(precropped_image, squeeze_dims=[0])
		  cropped_image = tf.random_crop(precropped_image_3d,
		                                 [MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH,
		                                  MODEL_INPUT_DEPTH])
		  if flip_left_right:
		    flipped_image = tf.image.random_flip_left_right(cropped_image)
		  else:
		    flipped_image = cropped_image
		  brightness_min = 1.0 - (random_brightness / 100.0)
		  brightness_max = 1.0 + (random_brightness / 100.0)
		  brightness_value = tf.random_uniform(tensor_shape.scalar(),
		                                       minval=brightness_min,
		                                       maxval=brightness_max)
		  distort_result = tf.multiply(flipped_image, brightness_value)
		  #distort_result = tf.expand_dims(brightened_image, 0, name='DistortResult')

		  self.distortion_image_data_input_placeholder = decoded_image_as_float
		  self.distort_image_data_operation = distort_result

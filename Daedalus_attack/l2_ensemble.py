import os
import sys
# supress tensorflow logging other than errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.insert(0, '..')

from keras import backend as K
import numpy as np
import random as rd
import tensorflow as tf
from tensorflow.python import debug as tf_debug
from keras.models import Model
from keras import losses

from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Dense, Dropout, Activation, Flatten

from keras.models import load_model
from keras.callbacks import EarlyStopping

from keras_retinanet import models as retina_models
from YOLOv3.model.yolo_model import YOLO
import cv2
import matplotlib.pyplot as plt
from skimage import io
import time

# Parameter settings:
GPU_ID = 0							# which gpu to used
ATTACK_MODE = 'all'					# select attack mode from 'all', 'most', 'least' and 'single';
ATTACK_CLASS = None					# select the class to attack in 'single' mode
CONFIDENCE = 0.3					# the confidence of attack
EXAMPLE_NUM = 10					# total number of adversarial example to generate.
BATCH_SIZE = 1						# number of adversarial example generated in each batch

BINARY_SEARCH_STEPS = 5     		# number of times to adjust the constsant with binary search
INITIAL_consts = 1e1        		# the initial constsant c to pick as a first guess
CLASS_NUM = 80						# 80 for COCO dataset
MAX_ITERATIONS = 10000      		# number of iterations to perform gradient descent
ABORT_EARLY = True          		# if we stop improving, abort gradient descent early
LEARNING_RATE = 1e-2        		# larger values converge faster to less accurate results
IMAGE_SHAPE = (416, 416, 3)         # input image shape
SAVE_PATH = 'adv_examples/L2/f3/{0}/'.format(ATTACK_MODE)
# select GPU to use
os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(GPU_ID)

def load_yolov3():
	return YOLO(0.5, 0.5)

def load_yolo():
	return YOLO(0.5, 0.5)

def load_retinanet():
	model_path = os.path.join('model', 'resnet50_coco_best_v2.1.0.h5')
	# load retinanet model
	oracle = retina_models.load_model(model_path, backbone_name='resnet50')
	oracle.layers.pop()
	oracle.outputs = [oracle.layers[-2].output, oracle.layers[-1].output] #remove nms from original model
	oracle.layers[-1].outbound_nodes = []
	oracle.summary()
	return oracle

def process_image(img):
	"""
	Resize, reduce and expand image.
	# Argument:
		img: original image.

	# Returns
		image: ndarray(64, 64, 3), processed image.
	"""
	image = cv2.resize(img, (416, 416),
					   interpolation=cv2.INTER_CUBIC)
	image = np.array(image, dtype='float32')
	image /= 255.
	image = np.expand_dims(image, axis=0)
	return image


def process_yolo_output(out, anchors, mask):
	"""
	Tensor op: Process output features.
	# Arguments
		out - tensor (N, S, S, 3, 4+1+80), output feature map of yolo.
		anchors - List, anchors for box.
		mask - List, mask for anchors.
	# Returns
		boxes - tensor (N, S, S, 3, 4), x,y,w,h for per box.
		box_confidence - tensor (N, S, S, 3, 1), confidence for per box.
		box_class_probs - tensor (N, S, S, 3, 80), class probs for per box.
	"""
	batchsize, grid_h, grid_w, num_boxes = map(int, out.shape[0:4])

	box_confidence = tf.sigmoid(out[..., 4:5], name='objectness')  # (N, S, S, 3, 1)
	box_class_probs = tf.sigmoid(out[..., 5:], name='class_probs')  # (N, S, S, 3, 80)

	anchors = np.array([anchors[i] for i in mask]) # Dimension of the used three anchor boxes [[x,x], [x,x], [x,x]].
	# duplicate to shape (batch, height, width, num_anchors, box_params).
	anchors = np.repeat(anchors[np.newaxis, :, :], grid_w, axis=0)          # (S, 3, 2)
	anchors = np.repeat(anchors[np.newaxis, :, :, :], grid_h, axis=0)       # (S, S, 3, 2)
	anchors = np.repeat(anchors[np.newaxis, :, :, :, :], batchsize, axis=0) # (N, S, S, 3, 2)
	anchors_tensors = tf.constant(anchors, dtype=tf.float32, name='anchor_tensors')

	box_xy = tf.sigmoid(out[..., 0:2], name='box_xy') # (N, S, S, 3, 2)
	box_wh = tf.identity(tf.exp(out[..., 2:4]) * anchors_tensors, name='box_wh') # (N, S, S, 3, 2)

	col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
	row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

	col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
	row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
	grid = np.concatenate((col, row), axis=-1) #(13, 13, 3, 2)
	grid_batch = np.repeat(grid[np.newaxis, :, :, :, :], batchsize, axis=0)
	box_xy += grid_batch
	box_xy /= (grid_w, grid_h)
	box_wh /= (416, 416)
	box_xy -= (box_wh / 2.)

	# boxes -> (N, S, S, 3, 4)
	boxes = tf.concat([box_xy, box_wh], axis=-1)
	boxes = tf.reshape(boxes, [batchsize, -1, boxes.shape[-2], boxes.shape[-1]], name='boxes') #(N, S*S, 3, 4)
	# box_confidence -> (N, S, S, 3, 1) or 26 or 52
	# box_class_probs -> (N, S, S, 3, 80)
	box_confidence = tf.reshape(box_confidence, [batchsize,
												 -1,
												 box_confidence.shape[-2],
												 box_confidence.shape[-1]], name='box_confidence')
	box_class_probs = tf.reshape(box_class_probs, [batchsize,
												   -1,
												   box_class_probs.shape[-2],
												   box_class_probs.shape[-1]], name='class_probs')
	return boxes, box_confidence, box_class_probs


def process_output(raw_outs):
	"""
	Tensor op: Extract b, c, and s from raw outputs.
	# Args:
		raw_outs - Yolo raw output tensor list [(N, 13, 13, 3, 85), (N, 26, 26, 3, 85), (N, 26, 26, 3, 85)].
	# Returns:
		boxes - Tensors. (N, 3549, 3, 4), classes: (N, 3549, 3, 1), scores: (N, 3549, 3, 80)
	"""
	masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	anchors = [[10, 13], [16, 30], [33, 23], 
			   [30, 61], [62, 45], [59, 119], 
			   [116, 90], [156, 198], [373, 326]]
	boxes, objecness, scores = [], [], []

	for out, mask in zip(raw_outs, masks):
		# out -> (N, 13, 13, 3, 85)
		# mask -> one of the masks
		# boxes (N, 13X13, 3, 4), box_confidence (N, 13X13, 3, 1)
		# box_class_probs (13X13, 3, 80) | 26 X 26 |
		b, c, s = process_yolo_output(out, anchors, mask)
		if boxes == []:
			boxes = b
			objecness = c
			scores = s
		else:
			boxes = tf.concat([boxes, b], 1, name='xywh') 
			objecness = tf.concat([objecness, c], 1, name='objectness')
			scores = tf.concat([scores, s], 1, name='class_probs')
	return boxes, objecness, scores

class Daedalus:
	"""
	Daedalus adversarial example generator based on the Yolo v3 model.
	"""
	def __init__(self, sess, models, target_class=ATTACK_CLASS, attack_mode=ATTACK_MODE, img_shape=IMAGE_SHAPE,
				 batch_size=BATCH_SIZE, confidence=CONFIDENCE, learning_rate=LEARNING_RATE, binary_search_steps=BINARY_SEARCH_STEPS,
				 max_iterations=MAX_ITERATIONS, abort_early=ABORT_EARLY, initial_consts=INITIAL_consts, boxmin=0, boxmax=1):

		# self.sess = tf_debug.LocalCLIDebugWrapperSession(sess)
		self.sess = sess
		self.LEARNING_RATE = learning_rate
		self.MAX_ITERATIONS = max_iterations
		self.BINARY_SEARCH_STEPS = binary_search_steps
		self.ABORT_EARLY = abort_early
		self.initial_consts = initial_consts
		self.batch_size = batch_size
		self.repeat = binary_search_steps >= 6
		self.yolo3 = models[0]
		self.yolo = models[1]
		self.retinanet = models[2]
		self.confidence = confidence
		self.img_dimension = img_shape[0]
		self.target_class = target_class
		self.attack_mode = attack_mode

		def select_class(target_class, boxes, objectness, box_scores, mode='all'):
			box_classes = tf.cast(tf.argmax(box_scores, axis=-1), tf.int32, name='box_classes')
			class_counts = tf.bincount(box_classes)
			print(class_counts)
			if mode == 'all':
				selected_boxes = tf.reshape(boxes, [BATCH_SIZE, -1, 4])
				selected_scores = tf.reshape(box_scores, [BATCH_SIZE, -1, CLASS_NUM])
				if objectness == None:
					return selected_boxes, None, selected_scores
				selected_objectness = tf.reshape(objectness, [BATCH_SIZE, -1, 1])
				return selected_boxes, selected_objectness, selected_scores
			elif mode == 'most':
				selected_cls = tf.argmax(class_counts)
			elif mode == 'least':
				class_counts = tf.where(tf.equal(class_counts,0), int(1e6)*tf.ones_like(class_counts, dtype=tf.int32), class_counts)
				selected_cls = tf.argmin(class_counts)
			elif mode == 'single':
				file = 'data/coco_classes.txt'
				with open(file) as f:
					class_names = f.readlines()
				class_names = [c.strip() for c in class_names]
				selected_cls = class_names.index(target_class)
			selected_cls = tf.cast(selected_cls, tf.int32)  
			index = tf.equal(box_classes, selected_cls)
			index = tf.cast(index, tf.int32)
			_, selected_boxes = tf.dynamic_partition(boxes, index, num_partitions=2, name='dynamic_partition')
			_, selected_scores = tf.dynamic_partition(box_scores, index, num_partitions=2, name='dynamic_partition')
			selected_boxes = tf.reshape(selected_boxes, [BATCH_SIZE, -1, 4])
			selected_scores = tf.reshape(selected_scores, [BATCH_SIZE, -1, CLASS_NUM])
			if objectness == None:
				return selected_boxes, None, selected_scores
			_, selected_objectness = tf.dynamic_partition(objectness, index, num_partitions=2, name='dynamic_partition')
			selected_objectness = tf.reshape(selected_objectness, [BATCH_SIZE, -1, 1])
			return selected_boxes, selected_objectness, selected_scores

		def yolov3_cg(images):
			# Get prediction from the model:
			outs = self.yolo3._yolo(images)
			# [(N, 13, 13, 3, 85), (N, 26, 26, 3, 85), (N, 52, 52, 3, 85)] to (N, 3549, 3, 4), (N, 3549, 3, 1), (N, 3549, 3, 80)
			boxes, objectness, classprobs = process_output(outs)
			boxes, objectness, classprobs = select_class(self.target_class, boxes, objectness, classprobs, mode=self.attack_mode)
			print(boxes, objectness, classprobs)
			self.yolo3bx = boxes[..., 0:1]
			self.yolo3by = boxes[..., 1:2]
			self.yolo3bw = boxes[..., 2:3]
			self.yolo3bh = boxes[..., 3:4]
			self.yolo3obj_scores = objectness
			self.yolo3class_probs = classprobs
			self.yolo3box_scores = tf.multiply(self.yolo3obj_scores, tf.reduce_max(self.yolo3class_probs, axis=-1, keepdims=True))
			return

		def retina_cg(images):
			caffe_imgs = images * 255.
			caffe_imgs = caffe_imgs[..., ::-1]
			caffe_offsets = np.concatenate([103.939*np.ones((batch_size, 416, 416, 1)),
									 	    116.779*np.ones((batch_size, 416, 416, 1)),
									        123.68*np.ones((batch_size, 416, 416, 1))], axis=-1)
			caffe_imgs = caffe_imgs - caffe_offsets
			boxes, classprobs = self.retinanet(images)
			boxes, _, classprobs = select_class(self.target_class, boxes, None, classprobs, mode=self.attack_mode)
			print(boxes, classprobs)
			self.retinax1 = boxes[..., 0:1]/self.img_dimension
			self.retinay1 = boxes[..., 1:2]/self.img_dimension
			self.retinax2 = boxes[..., 2:3]/self.img_dimension
			self.retinay2 = boxes[..., 3:4]/self.img_dimension
			self.retinabw = tf.math.abs(self.retinax2 - self.retinax1)
			self.retinabh = tf.math.abs(self.retinay1 - self.retinay2)
			self.retinaclass_probs = classprobs
			self.retinabox_scores = tf.reduce_max(self.retinaclass_probs, axis=-1, keepdims=True)
			return

		# the perturbation we're going to optimize:
		perturbations = tf.Variable(np.zeros((batch_size,
											  img_shape[0],
											  img_shape[1],
											  img_shape[2])), dtype=tf.float32, name='perturbations')
		# tf variables to sending data to tf:
		self.timgs = tf.Variable(np.zeros((batch_size,
										   img_shape[0],
										   img_shape[1],
										   img_shape[2])), dtype=tf.float32, name='self.timgs')
		self.consts = tf.Variable(np.zeros(batch_size), dtype=tf.float32, name='self.consts')

		# and here's what we use to assign them:
		self.assign_timgs = tf.placeholder(tf.float32, (batch_size,
														img_shape[0],
														img_shape[1],
														img_shape[2]))
		self.assign_consts = tf.placeholder(tf.float32, [batch_size])

		# Tensor operation: the resulting image, tanh'd to keep bounded from
		# boxmin to boxmax:
		self.boxmul = (boxmax - boxmin) / 2.
		self.boxplus = (boxmin + boxmax) / 2.
		self.newimgs = tf.tanh(perturbations + self.timgs) * self.boxmul + self.boxplus
		yolov3_cg(self.newimgs)
		retina_cg(self.newimgs)

		# Optimisation metrics:
		self.l2dist = tf.reduce_sum(tf.square(self.newimgs - (tf.tanh(self.timgs) * self.boxmul + self.boxplus)), [1, 2, 3])

		# Define DDoS losses: loss must be a tensor here!
		# Make the box confidence of all detections to be 1.
		self.loss1_1_x = tf.reduce_mean(tf.square(self.yolo3box_scores-1), [-2,-1]) + tf.reduce_mean(tf.square(self.retinabox_scores-1), [-2,-1])

		# Minimising the size of all bounding box.
		self.f3 = tf.reduce_mean(tf.square(tf.multiply(self.yolo3bw, self.yolo3bh)), [-2, -1]) + 1e3*tf.reduce_mean(tf.square(tf.multiply(self.retinabw, self.retinabh)), [-2, -1])

		# add two loss terms together
		self.loss_adv = self.loss1_1_x + self.f3
		self.loss1 = tf.reduce_mean(self.consts * self.loss_adv)
		self.loss2 = tf.reduce_mean(self.l2dist)
		self.loss = self.loss1 + self.loss2

		# Setup the adam optimizer and keep track of variables we're creating
		start_vars = set(x.name for x in tf.global_variables())
		optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
		self.train = optimizer.minimize(self.loss, var_list=[perturbations])
		end_vars = tf.global_variables()
		new_vars = [x for x in end_vars if x.name not in start_vars]

		# these are the variables to initialize when we run
		self.setup = []
		self.setup.append(self.timgs.assign(self.assign_timgs))
		self.setup.append(self.consts.assign(self.assign_consts))
		self.init = tf.variables_initializer(var_list=[perturbations] + new_vars)

	def attack_batch(self, imgs):
		"""
		Run the attack on a batch of images and labels.
		"""

		def check_success(loss, init_loss):
			"""
			Check if the initial loss value has been reduced by 'self.confidence' percent
			"""
			return loss <= init_loss * (1 - self.confidence)

		batch_size = self.batch_size

		# convert images to arctanh-space
		imgs = np.arctanh((imgs - self.boxplus) / self.boxmul * 0.999999)

		# set the lower and upper bounds of the constsant.
		lower_bound = np.zeros(batch_size)
		consts = np.ones(batch_size) * self.initial_consts
		upper_bound = np.ones(batch_size) * 1e10

		# store the best l2, score, and image attack
		o_bestl2 = [1e10] * batch_size
		o_bestloss = [1e10] * batch_size
		o_bestattack = [np.zeros(imgs[0].shape)] * batch_size

		for outer_step in range(self.BINARY_SEARCH_STEPS):
			# completely reset adam's internal state.
			self.sess.run(self.init)

			# take in the current data batch.
			batch = imgs[:batch_size]

			# cache the current best l2 and score.
			bestl2 = [1e10] * batch_size
			# bestconfidence = [-1]*batch_size
			bestloss = [1e10] * batch_size

			# The last iteration (if we run many steps) repeat the search once.
			if self.repeat and outer_step == self.BINARY_SEARCH_STEPS - 1:
				consts = upper_bound

			# set the variables so that we don't have to send them over again.
			self.sess.run(self.setup, {self.assign_timgs: batch,
									   self.assign_consts: consts})

			obj_grads = tf.gradients(self.loss1_1_x, self.newimgs)
			print('objectness gradients:', sess.run(obj_grads))
			loss_grads = tf.gradients(self.f3, self.newimgs)
			print('loss gradients:', sess.run(loss_grads))

			# start gradient descent attack
			print('adjust c to:', sess.run(self.consts))
			init_loss = sess.run(self.loss)
			init_adv_losses = sess.run(self.loss_adv)
			prev = init_loss * 1.1
			for iteration in range(self.MAX_ITERATIONS):
				# perform the attack on a single example
				_, l, l2s, l1s, nimgs, c = self.sess.run([self.train, self.loss, self.l2dist, self.loss_adv, self.newimgs, self.consts])
				# print out the losses every 10%
				if iteration % (self.MAX_ITERATIONS // 10) == 0:
					print('===iteration:', iteration, '===')
					print('attacked box number:', sess.run(self.yolo3bw).shape, sess.run(self.retinabw).shape)
					print('loss values of box confidence and dimension:', sess.run([self.loss1_1_x, self.f3]))
					print('adversarial losses:', l1s)
					print('distortions:', l2s)

				# check if we should abort search if we're getting nowhere.
				if self.ABORT_EARLY and iteration % (self.MAX_ITERATIONS // 10) == 0:
					if l > prev * .9999:
						break
					prev = l

				# update the best result found so far
				for e, (l1, l2, ii) in enumerate(zip(l1s, l2s, nimgs)):
					if l2 < bestl2[e] and check_success(l1, init_adv_losses[e]):
						bestl2[e] = l2
						bestloss[e] = l1
					if l2 < o_bestl2[e] and check_success(l1, init_adv_losses[e]):
						o_bestl2[e] = l2
						o_bestloss[e] = l1
						o_bestattack[e] = ii

			# adjust the constsant as needed
			for e in range(batch_size):
				if check_success(l1s[e], init_adv_losses[e]):
					# success, divide consts by two
					upper_bound[e] = min(upper_bound[e], consts[e])
					if upper_bound[e] < 1e9:
						consts[e] = (lower_bound[e] + upper_bound[e]) / 2
				else:
					# failure, either multiply by 10 if no solution found yet
					#          or do binary search with the known upper bound
					lower_bound[e] = max(lower_bound[e], consts[e])
					if upper_bound[e] < 1e9:
						consts[e] = (lower_bound[e] + upper_bound[e]) / 2
					else:
						consts[e] *= 10
		# return the best solution found
		o_bestl2 = np.array(o_bestl2)
		return o_bestattack, o_bestl2


	def attack(self, imgs):
		"""
		Perform the L_2 attack on the given images for the given targets.
		If self.targeted is true, then the targets represents the target labels.
		If self.targeted is false, then targets are the original class labels.
		"""
		r = []
		ds = []
		print('go up to', len(imgs))
		for i in range(0, len(imgs), self.batch_size):
			print('tick', i)
			X_adv, dists = self.attack_batch(imgs[i:i + self.batch_size])
			path = SAVE_PATH+'ensemble/{0} confidence'.format(self.confidence)
			if not os.path.exists(path):
				os.makedirs(path)
			np.save(path+'/Distortions of images {0} to {1}.npy'.format(i, i+self.batch_size), dists)
			for j in range(len(X_adv)):
				io.imsave(path+'/Best example of {1} Distortion {2}.png'.format(self.confidence, i+j, dists[j]), X_adv[j])			
			r.extend(X_adv)
			ds.extend(dists)
		return np.array(r), np.array(ds)


if __name__ == '__main__':
	sess = tf.InteractiveSession()
	init = tf.global_variables_initializer()
	sess.run(init)
	ORACLEs = [load_yolov3(), load_yolo(), load_retinanet()]  # The auguments do not matter.
	attacker = Daedalus(sess, ORACLEs)

	X_test = []
	for (root, dirs, files) in os.walk('../Datasets/COCO/val2017/'):
		if files:
			for f in files:
				print(f)
				path = os.path.join(root, f)
				image = cv2.imread(path)
				image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # RGB
				image = process_image(image)
				#image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
				X_test.append(image)
				EXAMPLE_NUM -= 1
				if EXAMPLE_NUM == 0:
					break
	X_test = np.concatenate(X_test, axis=0)
	
	start = time.time()
	X_adv, distortions = attacker.attack(X_test)
	end = time.time()
	print('time: {0:.2f}s'.format((end - start)*0.2))
	f = open('f2 runtime.txt', 'a')
	f.write('time: {0:.2f}s\n'.format((end - start)*0.2))
	f.close()
	np.savez(SAVE_PATH+'ensemble/{} confidence/Daedalus example batch.npz'.format(CONFIDENCE), X_adv=X_adv, distortions=distortions)
	writer = tf.summary.FileWriter("log", sess.graph)
	writer.close()

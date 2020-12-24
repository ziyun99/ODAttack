#!/usr/bin/env python3.6
#
# Copyright (C) 2019 Intel Corporation
#
# SPDX-License-Identifier: BSD-3-Clause
#
import os
import sys
import time
import math
import logging
import numpy as np
import tensorflow as tf

from functools import partial
from PIL import Image

from shapeshifter import parse_args
from shapeshifter import load_and_tile_images
from shapeshifter import create_model
from shapeshifter import create_textures
from shapeshifter import create_attack
from shapeshifter import create_evaluation
from shapeshifter import batch_accumulate
from shapeshifter import plot

import argparse
import matplotlib.pyplot as plt

from object_detection.utils import config_util
from object_detection.utils import label_map_util as label_map_util
from object_detection.utils import variables_helper
from object_detection.utils import visualization_utils

from object_detection.builders import model_builder
from object_detection.utils.object_detection_evaluation import ObjectDetectionEvaluator
from object_detection.core.standard_fields import InputDataFields, DetectionResultFields

class Shapeshifter(object):
    def __init__(self, config, argrs):
        self.config = self.load_config(config)
        self.setup_logging()
        # self.load_attack()
        return

    def load_config(self, config):
        # Create tuples from -min -max self.config
        for key in list(vars(config)):
            if key.endswith('_min'):
                key = key[:-4]
                setattr(config, key + '_range', (getattr(config, key + '_min'), getattr(config, key + '_max')))

        # Parse self.config.model directory
        config.model_config = config.model + '/pipeline.config'
        config.model_checkpoint = config.model + '/model.ckpt'
        config.model_labels = config.model + '/label_map.pbtxt'

        # Read labels and set target class and victim class ids
        config.label_map = label_map_util.get_label_map_dict(config.model_labels, use_display_name=True)
        config.category_index = label_map_util.create_category_index_from_labelmap(config.model_labels)
        config.categories = list(config.category_index.values())

        config.target_class = config.label_map[config.target]
        config.victim_class = config.label_map[config.victim]
        return config
    
    def setup_logging(self):
        # Setup logging
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.ERROR)
        tf.logging.set_verbosity(tf.logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

        self.log = logging.getLogger('shapeshifter')
        self.log.setLevel(logging.DEBUG if self.config.verbose else logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s: %(levelname)s %(name)s: %(message)s')
        self.handler = logging.StreamHandler()
        self.handler.setFormatter(self.formatter)
        self.log.addHandler(self.handler)
        self.log.propagate = False

    def attack(self):
        # Set seeds from the start
        if self.config.seed:
            self.log.debug("Setting seed")
            np.random.seed(self.config.seed)
            tf.set_random_seed(self.config.seed)

        # Load textures, backgrounds and objects (and their masks)
        self.log.debug("Loading backgrounds, textures, textures masks, objects, and objects masks")
        backgrounds = load_and_tile_images(self.config.backgrounds)

        textures = load_and_tile_images(self.config.textures)
        textures = textures[:, :, :, :3]
        textures_masks = (load_and_tile_images(self.config.textures_masks)[:, :, :, :1] >= 0.5).astype(np.float32)
        assert(textures.shape[:3] == textures_masks.shape[:3])

        objects = load_and_tile_images(self.config.objects)
        objects_masks = (objects[:, :, :, 3] >= 0.5).astype(np.float32)
        objects = objects[:, :, :, :3]
        assert(objects.shape[:3] == objects_masks.shape[:3])

        # Create test data
        generate_data_partial = partial(generate_data,
                                        backgrounds=backgrounds,
                                        objects=objects,
                                        objects_masks=objects_masks,
                                        objects_class=self.config.target_class,
                                        objects_transforms={'yaw_range': self.config.object_yaw_range,
                                                            'yaw_bins': self.config.object_yaw_bins,
                                                            'yaw_fn': self.config.object_yaw_fn,

                                                            'pitch_range': self.config.object_pitch_range,
                                                            'pitch_bins': self.config.object_pitch_bins,
                                                            'pitch_fn': self.config.object_pitch_fn,

                                                            'roll_range': self.config.object_roll_range,
                                                            'roll_bins': self.config.object_roll_bins,
                                                            'roll_fn': self.config.object_roll_fn,

                                                            'x_range': self.config.object_x_range,
                                                            'x_bins': self.config.object_x_bins,
                                                            'x_fn': self.config.object_x_fn,

                                                            'y_range': self.config.object_y_range,
                                                            'y_bins': self.config.object_y_bins,
                                                            'y_fn': self.config.object_y_fn,

                                                            'z_range': self.config.object_z_range,
                                                            'z_bins': self.config.object_z_bins,
                                                            'z_fn': self.config.object_z_fn},
                                        textures_transforms={'yaw_range': self.config.texture_yaw_range,
                                                            'yaw_bins': self.config.texture_yaw_bins,
                                                            'yaw_fn': self.config.texture_yaw_fn,

                                                            'pitch_range': self.config.texture_pitch_range,
                                                            'pitch_bins': self.config.texture_pitch_bins,
                                                            'pitch_fn': self.config.texture_pitch_fn,

                                                            'roll_range': self.config.texture_roll_range,
                                                            'roll_bins': self.config.texture_roll_bins,
                                                            'roll_fn': self.config.texture_roll_fn,

                                                            'x_range': self.config.texture_x_range,
                                                            'x_bins': self.config.texture_x_bins,
                                                            'x_fn': self.config.texture_x_fn,

                                                            'y_range': self.config.texture_y_range,
                                                            'y_bins': self.config.texture_y_bins,
                                                            'y_fn': self.config.texture_y_fn,

                                                            'z_range': self.config.texture_z_range,
                                                            'z_bins': self.config.texture_z_bins,
                                                            'z_fn': self.config.texture_z_fn},
                                        seed=self.config.seed)

        # Create adversarial textures, composite them on object, and pass composites into model. Finally, create summary statistics.
        self.log.debug("Creating perturbable textures")
        textures_var_, textures_ = create_textures(textures, 1.0, # initial_texture doesn't really matter
                                                use_spectral=self.config.spectral,
                                                soft_clipping=self.config.soft_clipping)

        self.log.debug("Creating composited input images")
        input_images_ = create_composited_images(self.config.batch_size, textures_, textures_masks)

        self.log.debug("Creating object detection model")
        predictions, detections, losses = create_model(input_images_, self.config.model_config, self.config.model_checkpoint, is_training=True)

        self.log.debug("Creating attack losses")
        victim_class_, target_class_, losses_summary_ = create_attack(textures_, textures_var_, predictions, losses,
                                                                    optimizer_name=self.config.optimizer, clip=self.config.sign_gradients)

        self.log.debug("Creating evaluation metrics")
        metrics_summary_, texture_summary_ = create_evaluation(victim_class_, target_class_,
                                                            textures_, textures_masks, input_images_, detections)

        summaries_ = tf.summary.merge([losses_summary_, metrics_summary_])

        global_init_op_ = tf.global_variables_initializer()
        local_init_op_ = tf.local_variables_initializer()

        # Create tensorboard file writer for train and test evaluations
        self.saver = tf.train.Saver([textures_var_, tf.train.get_global_step()])
        self.train_writer = None
        self.test_writer = None

        if self.config.logdir is not None:
            self.log.debug(f"Tensorboard logging: {self.config.logdir}")
            os.makedirs(self.config.logdir, exist_ok=True)

            arguments_summary_ = tf.summary.text('Arguments', tf.constant('```' + ' '.join(sys.argv[1:]) + '```'))
            # TODO: Save argparse

            graph = None
            if self.config.save_graph:
                self.log.debug("Graph will be saved to tensorboard")
                graph = tf.get_default_graph()

            self.train_writer = tf.summary.FileWriter(self.config.logdir + '/train', graph=graph)
            self.test_writer = tf.summary.FileWriter(self.config.logdir + '/test')

            # Find existing checkpoint
            os.makedirs(self.config.logdir + '/checkpoints', exist_ok=True)
            checkpoint_path = tf.train.latest_checkpoint(self.config.logdir + '/checkpoints')
            self.config.checkpoint = checkpoint_path

        # Create session
        self.log.debug("Creating session")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        sess = tf.Session(config=config)
        sess.run(global_init_op_)
        sess.run(local_init_op_)

        # Set initial texture
        if self.config.checkpoint is not None:
            self.log.debug(f"Restoring from checkpoint: {self.config.checkpoint}")
            self.saver.restore(sess, self.config.checkpoint)
        else:
            if self.config.gray_start:
                self.log.debug("Setting texture to gray")
                textures = np.zeros_like(textures) + 128/255

            if self.config.random_start > 0:
                self.log.debug(f"Adding uniform random perturbation texture with at most {self.config.random_start}/255 per pixel")
                textures = textures + np.random.randint(size=textures.shape, low=-self.config.random_start, high=self.config.random_start)/255

            sess.run('project_op', { 'textures:0': textures  })

        # Get global step
        step = sess.run('global_step:0')

        if self.train_writer is not None:
            self.log.debug("Running arguments summary")
            summary = sess.run(arguments_summary_)

            self.train_writer.add_summary(summary, step)
            self.test_writer.add_summary(summary, step)

        loss_tensors = ['total_loss:0',
                        'total_rpn_cls_loss:0', 'total_rpn_loc_loss:0',
                        'total_rpn_foreground_loss:0', 'total_rpn_background_loss:0',
                        'total_rpn_cw_loss:0',
                        'total_box_cls_loss:0', 'total_box_loc_loss:0',
                        'total_box_target_loss:0', 'total_box_victim_loss:0',
                        'total_box_target_cw_loss:0', 'total_box_victim_cw_loss:0',
                        'total_sim_loss:0',
                        'grad_l2:0', 'grad_linf:0']

        metric_tensors = ['proposal_average_precision:0', 'victim_average_precision:0', 'target_average_precision:0']

        output_tensors = loss_tensors + metric_tensors

        self.log.info('global_step [%s]', ', '.join([tensor.replace(':0', '').replace('total_', '') for tensor in output_tensors]))

        test_feed_dict = { 'learning_rate:0': self.config.learning_rate,
                        'momentum:0': self.config.momentum,
                        'decay:0': self.config.decay,

                        'rpn_iou_thresh:0': self.config.rpn_iou_threshold,
                        'rpn_cls_weight:0': self.config.rpn_cls_weight,
                        'rpn_loc_weight:0': self.config.rpn_loc_weight,
                        'rpn_foreground_weight:0': self.config.rpn_foreground_weight,
                        'rpn_background_weight:0': self.config.rpn_background_weight,
                        'rpn_cw_weight:0': self.config.rpn_cw_weight,
                        'rpn_cw_conf:0': self.config.rpn_cw_conf,

                        'box_iou_thresh:0': self.config.box_iou_threshold,
                        'box_cls_weight:0': self.config.box_cls_weight,
                        'box_loc_weight:0': self.config.box_loc_weight,
                        'box_target_weight:0': self.config.box_target_weight,
                        'box_victim_weight:0': self.config.box_victim_weight,
                        'box_target_cw_weight:0': self.config.box_target_cw_weight,
                        'box_target_cw_conf:0': self.config.box_target_cw_conf,
                        'box_victim_cw_weight:0': self.config.box_victim_cw_weight,
                        'box_victim_cw_conf:0': self.config.box_victim_cw_conf,

                        'sim_weight:0': self.config.sim_weight,

                        'victim_class:0': self.config.victim_class,
                        'target_class:0': self.config.target_class }

    # def attack(self):
        # Keep attacking until CTRL+C. The only issue is that we may be in the middle of some operation.
        try:
            self.log.debug("Entering attacking loop (use ctrl+c to exit)")
            while True:
                # Run summaries as necessary
                if self.config.logdir and step % self.config.save_checkpoint_every == 0:
                    self.log.debug("Saving checkpoint")
                    self.saver.save(sess, self.config.logdir + '/checkpoints/texture', global_step=step, write_meta_graph=False, write_state=True)

                if step % self.config.save_texture_every == 0 and self.test_writer is not None:
                    self.log.debug("Writing texture summary")
                    test_texture = sess.run(texture_summary_)
                    self.test_writer.add_summary(test_texture, step)

                if step % self.config.save_test_every == 0:
                    self.log.debug("Runnning test summaries")
                    start_time = time.time()

                    sess.run(local_init_op_)
                    batch_accumulate(sess, test_feed_dict,
                                    self.config.test_batch_size, self.config.batch_size,
                                    generate_data_partial,
                                    detections, predictions, self.config.categories)

                    end_time = time.time()
                    self.log.debug(f"Loss accumulation took {end_time - start_time} seconds")

                    test_output = sess.run(output_tensors, test_feed_dict)
                    self.log.info('test %d %s', step, test_output)

                    if self.test_writer is not None:
                        self.log.debug("Writing test summaries")
                        test_summaries = sess.run(summaries_, test_feed_dict)
                        self.test_writer.add_summary(test_summaries, step)

                # Create train feed_dict
                train_feed_dict = test_feed_dict.copy()

                train_feed_dict['image_channel_multiplicative_noise:0'] = self.config.image_multiplicative_channel_noise_range
                train_feed_dict['image_channel_additive_noise:0'] = self.config.image_additive_channel_noise_range
                train_feed_dict['image_pixel_multiplicative_noise:0'] = self.config.image_multiplicative_pixel_noise_range
                train_feed_dict['image_pixel_additive_noise:0'] = self.config.image_additive_pixel_noise_range
                train_feed_dict['image_gaussian_noise_stddev:0'] = self.config.image_gaussian_noise_stddev_range

                train_feed_dict['texture_channel_multiplicative_noise:0'] = self.config.texture_multiplicative_channel_noise_range
                train_feed_dict['texture_channel_additive_noise:0'] = self.config.texture_additive_channel_noise_range
                train_feed_dict['texture_pixel_multiplicative_noise:0'] = self.config.texture_multiplicative_pixel_noise_range
                train_feed_dict['texture_pixel_additive_noise:0'] = self.config.texture_additive_pixel_noise_range
                train_feed_dict['texture_gaussian_noise_stddev:0'] = self.config.texture_gaussian_noise_stddev_range

                # Zero out gradient accumulation, losses, and metrics, then accumulate batches
                self.log.debug("Starting gradient accumulation...")
                start_time = time.time()


                sess.run(local_init_op_)
                batch_accumulate(sess, train_feed_dict,
                                self.config.train_batch_size, self.config.batch_size,
                                generate_data_partial,
                                detections, predictions, self.config.categories)

                end_time = time.time()
                self.log.debug(f"Gradient accumulation took {end_time - start_time} seconds")

                train_output = sess.run(output_tensors, train_feed_dict)
                self.log.info('train %d %s', step, train_output)

                if step % self.config.save_train_every == 0 and self.train_writer is not None:
                    self.log.debug("Writing train summaries")
                    train_summaries = sess.run(summaries_, test_feed_dict)
                    self.train_writer.add_summary(train_summaries, step)

                # Update textures and project texture to feasible set
                # TODO: We can probably run these together but probably need some control dependency
                self.log.debug("Projecting attack")
                sess.run('attack_op', train_feed_dict)
                sess.run('project_op')
                step = sess.run('global_step:0')

        except KeyboardInterrupt:
            self.log.warn('Interrupted')

        finally:
            if self.test_writer is not None:
                self.test_writer.close()

            if self.train_writer is not None:
                self.train_writer.close()

            if sess is not None:
                sess.close()

def create_composited_images(batch_size, textures, masks, interpolation='BILINEAR'):
    backgrounds_ = tf.placeholder(tf.float32, [batch_size, None, None, 3], name='backgrounds')
    transforms_ = tf.placeholder(tf.float32, shape=[batch_size, 8], name='transforms')

    texture_channel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='texture_channel_multiplicative_noise')
    texture_channel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='texture_channel_additive_noise')

    texture_pixel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='texture_pixel_multiplicative_noise')
    texture_pixel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='texture_pixel_additive_noise')

    texture_gaussian_noise_stddev_ = tf.placeholder_with_default([0., 0.], [2], name='texture_gaussian_noise_stddev')

    image_channel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='image_channel_multiplicative_noise')
    image_channel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='image_channel_additive_noise')

    image_pixel_multiplicative_noise_ = tf.placeholder_with_default([1., 1.], [2], name='image_pixel_multiplicative_noise')
    image_pixel_additive_noise_ = tf.placeholder_with_default([0., 0.], [2], name='image_pixel_additive_noise')

    image_gaussian_noise_stddev_ = tf.placeholder_with_default([0., 0.], [2], name='image_gaussian_noise_stddev')

    backgrounds_shape_ = tf.shape(backgrounds_)

    transformed_textures_ = textures
    transformed_masks_ = masks

    # Add noise to textures
    # TODO: Add blur to transformed textures
    transformed_textures_ = transformed_textures_ * tf.random_uniform([3], texture_channel_multiplicative_noise_[0], texture_channel_multiplicative_noise_[1])
    transformed_textures_ = transformed_textures_ + tf.random_uniform([3], texture_channel_additive_noise_[0], texture_channel_additive_noise_[1])

    transformed_textures_ = transformed_textures_ * tf.random_uniform([], texture_pixel_multiplicative_noise_[0], texture_pixel_multiplicative_noise_[1])
    transformed_textures_ = transformed_textures_ + tf.random_uniform([], texture_pixel_additive_noise_[0], texture_pixel_additive_noise_[1])

    transformed_textures_ = transformed_textures_ + tf.random_normal(transformed_textures_.shape, stddev=tf.random_uniform([], texture_gaussian_noise_stddev_[0], texture_gaussian_noise_stddev_[1]))

    #transformed_textures_ = tf.clip_by_value(transformed_textures_, 0.0, 1.0)

    # Center texture and mask in image the same size as backgrounds
    target_height = backgrounds_shape_[1]
    target_width = backgrounds_shape_[2]
    offset_height = (target_height - textures.shape[1]) // 2
    offset_width = (target_width - textures.shape[2]) // 2

    transformed_textures_ = tf.image.pad_to_bounding_box(transformed_textures_, offset_height, offset_width, target_height, target_width)
    transformed_masks_ = tf.image.pad_to_bounding_box(transformed_masks_, offset_height, offset_width, target_height, target_width)

    # Transform texture and mask ensure output remains the same size as backgrounds
    transformed_textures_ = tf.contrib.image.transform(transformed_textures_, transforms_, interpolation, backgrounds_shape_[1:3])
    transformed_masks_ = tf.contrib.image.transform(transformed_masks_, transforms_, interpolation, backgrounds_shape_[1:3])

    transformed_textures_ = tf.identity(transformed_textures_, name='transformed_textures')
    transformed_masks_ = tf.identity(transformed_masks_, name='transformed_masks')

    input_images_ = backgrounds_ * (1. - transformed_masks_) + transformed_textures_ * (transformed_masks_)

    # Add noise to image
    # TODO: Add blur to composite images
    input_images_ = input_images_ * tf.random_uniform([3], image_channel_multiplicative_noise_[0], image_channel_multiplicative_noise_[1])
    input_images_ = input_images_ + tf.random_uniform([3], image_channel_additive_noise_[0], image_channel_additive_noise_[1])

    input_images_ = input_images_ * tf.random_uniform([], image_pixel_multiplicative_noise_[0], image_pixel_multiplicative_noise_[1])
    input_images_ = input_images_ + tf.random_uniform([], image_pixel_additive_noise_[0], image_pixel_additive_noise_[1])

    input_images_ = input_images_ + tf.random_normal(tf.shape(input_images_), stddev=tf.random_uniform([], image_gaussian_noise_stddev_[0], image_gaussian_noise_stddev_[1]))

    #input_images_ = tf.clip_by_value(input_images_, 0.0, 1.0)

    input_images_ = tf.fake_quant_with_min_max_args(input_images_, min=0., max=1., num_bits=8)
    input_images_ = tf.identity(input_images_, name='input_images')

    return input_images_

def _apply_blur(image):
    image_ = tf.expand_dims(image, axis=0)

    kernel_ = tf.expand_dims(tf.expand_dims(tf.eye(3), 0), 0)

    # Horizontal blur with distance sampled from beta distribution
    horizontal_blur_ = tf.random_gamma([], horizontal_blur_alpha_)
    horizontal_blur_ = horizontal_blur_ / (horizontal_blur_ + tf.random_gamma([], horizontal_blur_beta_))
    horizontal_blur_ = tf.cast(horizontal_blur_ * horizontal_blur_max_, tf.int32) + 1

    horizontal_blur_kernel_ = tf.tile(kernel_, (1, horizontal_blur_, 1, 1))
    horizontal_blur_kernel_ = horizontal_blur_kernel_ / tf.cast(horizontal_blur_, tf.float32)

    image_ = tf.nn.conv2d(image_, horizontal_blur_kernel_, [1, 1, 1, 1], 'SAME')

    # Vertical blur with distance sampled from beta distribution
    vertical_blur_ = tf.random_gamma([], vertical_blur_alpha_)
    vertical_blur_ = vertical_blur_ / (vertical_blur_ + tf.random_gamma([], vertical_blur_beta_))
    vertical_blur_ = tf.cast(vertical_blur_ * vertical_blur_max_, tf.int32) + 1

    vertical_blur_kernel_ = tf.tile(kernel_, (vertical_blur_, 1, 1, 1))
    vertical_blur_kernel_ = vertical_blur_kernel_ / tf.cast(vertical_blur_, tf.float32)

    image_ = tf.nn.conv2d(image_, vertical_blur_kernel_, [1, 1, 1, 1], 'SAME')

    return image_[0]

def generate_data(count, backgrounds, objects, objects_masks, objects_class, objects_transforms, textures_transforms, seed=None):
    if seed:
        np.random.seed(seed)

    assert(objects.shape[0] == 1)
    assert(objects_masks.shape[0] == 1)

    backgrounds_count, output_height, output_width = backgrounds.shape[0:3]
    origin = (output_width // 2, output_height // 2)

    # Extract bounding boxes after transforming mask
    object_img = Image.fromarray((255*objects[0]).astype(np.uint8))
    mask_img = Image.fromarray((255*objects_masks[0]).astype(np.uint8))

    bboxes = []
    composited_backgrounds = []

    objects_transforms = generate_transforms(origin, count, **objects_transforms)

    for objects_transform in objects_transforms:
        target_left = (output_width - object_img.width) // 2
        target_upper = (output_height - object_img.height) // 2

        # Transform mask to background dimensions, then compute normalized bounding box
        transformed_mask_img = Image.new('L', (output_width, output_height), (0))
        transformed_mask_img.paste(mask_img, (target_left, target_upper), mask_img)
        transformed_mask_img = transformed_mask_img.transform((output_width, output_height), Image.PERSPECTIVE, objects_transform, Image.BILINEAR)

        # Composite object onto random background and transform
        transformed_object_img = Image.new('RGB', (output_width, output_height), (0, 0, 0))
        transformed_object_img.paste(object_img, (target_left, target_upper), mask_img)
        transformed_object_img = transformed_object_img.transform((output_width, output_height), Image.PERSPECTIVE, objects_transform, Image.BILINEAR)

        composited_background_img = Image.fromarray((255*backgrounds[np.random.choice(backgrounds_count)]).astype(np.uint8))
        composited_background_img.paste(transformed_object_img, (0, 0), transformed_mask_img)

        composited_background = np.array(composited_background_img) / 255
        composited_backgrounds.append(composited_background)

        # Compute normalized bounding box using transformed_mask
        transformed_mask = np.array(transformed_mask_img) / 255
        rows = np.any(transformed_mask[:, :] > 0.5, axis=0)
        cols = np.any(transformed_mask[:, :] > 0.5, axis=1)

        ymin = np.argmax(cols)
        ymax = output_height - np.argmax(cols[::-1]) - 1
        xmin = np.argmax(rows)
        xmax = output_width - np.argmax(rows[::-1]) - 1

        ymin /= output_height
        ymax /= output_height
        xmin /= output_width
        xmax /= output_width

        bbox = np.array([[ymin, xmin, ymax, xmax]], dtype=np.float32)

        bboxes.append(bbox)

    bboxes = np.array(bboxes)
    composited_backgrounds = np.array(composited_backgrounds)

    # Create texture transforms and compose them with background transforms
    textures_transforms = generate_transforms(origin, count, **textures_transforms)

    objects_transforms = np.concatenate([objects_transforms, np.ones((objects_transforms.shape[0], 1))], axis=1)
    textures_transforms = np.concatenate([textures_transforms, np.ones((textures_transforms.shape[0], 1))], axis=1)

    objects_transforms = np.reshape(objects_transforms, (-1, 3, 3))
    textures_transforms = np.reshape(textures_transforms, (-1, 3, 3))

    transforms = np.matmul(textures_transforms, objects_transforms)
    transforms = np.reshape(transforms, (-1, 9))
    transforms = transforms[:, :8]

    data = {'transforms:0': transforms,
            'backgrounds:0': composited_backgrounds,
            'groundtruth_boxes_%d:0': bboxes,
            'groundtruth_classes_%d:0': np.full(bboxes.shape[0:2], objects_class - 1),
            'groundtruth_weights_%d:0': np.ones(bboxes.shape[0:2])}

    return data

def generate_transforms(origin, count,
                        yaws=None, yaw_range=(0, 0), yaw_bins=100, yaw_fn=np.linspace,
                        pitchs=None, pitch_range=(0, 0), pitch_bins=100, pitch_fn=np.linspace,
                        rolls=None, roll_range=(0, 0), roll_bins=100, roll_fn=np.linspace,
                        xs=None, x_range=(0, 0), x_bins=100, x_fn=np.linspace,
                        ys=None, y_range=(0, 0), y_bins=100, y_fn=np.linspace,
                        zs=None, z_range=(0, 0), z_bins=100, z_fn=np.linspace):
     # Discretize ranges
    if yaws is None:
        yaws = yaw_fn(*yaw_range, num=yaw_bins)
    if pitchs is None:
        pitchs = pitch_fn(*pitch_range, num=pitch_bins)
    if rolls is None:
        rolls = roll_fn(*roll_range, num=roll_bins)
    if xs is None:
        xs = x_fn(*x_range, num=x_bins)
    if ys is None:
        ys = y_fn(*y_range, num=y_bins)
    if zs is None:
        zs = z_fn(*z_range, num=z_bins)

    # Choose randomly
    yaws = np.random.choice(yaws, count)
    pitchs = np.random.choice(pitchs, count)
    rolls = np.random.choice(rolls, count)
    xs = np.random.choice(xs, count)
    ys = np.random.choice(ys, count)
    zs = np.random.choice(zs, count)

    # Get transforms for each options
    transforms = []

    for yaw, pitch, roll, x, y, z in zip(yaws, pitchs, rolls, xs, ys, zs):
        transform = get_transform(origin, x_shift=x, y_shift=y, im_scale=z, rot_in_degrees=roll)
        transforms.append(transform)

    transforms = np.array(transforms).astype(np.float32)

    return transforms

# From: https://github.com/tensorflow/cleverhans/blob/master/examples/adversarial_patch/AdversarialPatch.ipynb
def get_transform(origin, x_shift, y_shift, im_scale, rot_in_degrees):
    """
    If one row of transforms is [a0, a1, a2, b0, b1, b2, c0, c1],
    then it maps the output point (x, y) to a transformed input point
    (x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k),
    where k = c0 x + c1 y + 1.
    The transforms are inverted compared to the transform mapping input points to output points.
    """
    rot = float(rot_in_degrees) / 90. * (math.pi/2)

    # Standard rotation matrix
    # (use negative rot because tf.contrib.image.transform will do the inverse)
    rot_matrix = np.array(
        [[math.cos(-rot), -math.sin(-rot)],
        [math.sin(-rot), math.cos(-rot)]]
    )

    # Scale it
    # (use inverse scale because tf.contrib.image.transform will do the inverse)
    inv_scale = 1. / im_scale
    xform_matrix = rot_matrix * inv_scale
    a0, a1 = xform_matrix[0]
    b0, b1 = xform_matrix[1]

    # At this point, the image will have been rotated around the top left corner,
    # rather than around the center of the image.
    #
    # To fix this, we will see where the center of the image got sent by our transform,
    # and then undo that as part of the translation we apply.
    #x_origin = float(width) / 2
    #y_origin = float(width) / 2
    x_origin, y_origin = origin

    x_origin_shifted, y_origin_shifted = np.matmul(
        xform_matrix,
        np.array([x_origin, y_origin]),
    )

    x_origin_delta = x_origin - x_origin_shifted
    y_origin_delta = y_origin - y_origin_shifted

    # Combine our desired shifts with the rotation-induced undesirable shift
    a2 = x_origin_delta - (x_shift/(2*im_scale))
    b2 = y_origin_delta - (y_shift/(2*im_scale))

    # Return these values in the order that tf.contrib.image.transform expects
    return np.array([a0, a1, a2, b0, b1, b2, 0, 0]).astype(np.float32)

if __name__ == '__main__':
    main()

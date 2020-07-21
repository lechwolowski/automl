# Lint as: python3
# Copyright 2020 Google Research. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""Mosaic data augmentation"""
import tensorflow as tf


def crop_image_and_box(image, size):
  limit = tf.shape(image) - size + 1
  offset = tf.random.uniform(
      [3], dtype=limit.dtype, minval=0, maxval=limit.dtype.max) % limit
  crop_image = tf.slice(image, offset, size)
  return crop_image, offset


def clip_box(box, clazz, min, max):
  ymin = tf.clip_by_value(box[:, 0], min[0], max[0])
  xmin = tf.clip_by_value(box[:, 1], min[1], max[1])
  ymax = tf.clip_by_value(box[:, 2], min[0], max[0])
  xmax = tf.clip_by_value(box[:, 3], min[1], max[1])
  cliped_boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)
  mask = tf.logical_or(
      tf.not_equal(xmax - xmin, 0), tf.not_equal(ymax - ymin, 0))
  final_boxes = tf.boolean_mask(cliped_boxes, mask, axis=0)
  final_classes = tf.boolean_mask(clazz, mask, axis=0)
  return final_boxes, final_classes


def calculate_boxes_offset(box, offset_y, offset_x):
  return tf.stack([
      box[:, 0] + offset_y, box[:, 1] + offset_x, box[:, 2] + offset_y,
      box[:, 3] + offset_x
  ],
                  axis=1)


def mosaic(images, boxes, classes, size):
  with tf.name_scope('mosaic'):
    y = tf.random.uniform([], 0.25, 0.75)
    x = tf.random.uniform([], 0.25, 0.75)

    temp_size = tf.cast(
        [tf.math.round(y * size[0]),
         tf.math.round(x * size[1]), 3], tf.int32)
    crop_image1, offset = crop_image_and_box(images[0], temp_size)
    box = calculate_boxes_offset(boxes[0],
                                 tf.cast(-offset[0] / size[0], tf.float32),
                                 tf.cast(-offset[1], tf.float32))
    boxes[0], classes[0] = clip_box(box, classes[0], [0, 0], [y, x])

    temp_size = tf.cast(
        [tf.math.round(size[0] - y * size[0]),
         tf.math.round(x * size[1]), 3], tf.int32)
    crop_image2, offset = crop_image_and_box(images[1], temp_size)
    box = calculate_boxes_offset(boxes[1],
                                 y - tf.cast(offset[0] / size[0], tf.float32),
                                 tf.cast(-offset[1] / size[1], tf.float32))
    boxes[1], classes[1] = clip_box(box, classes[1], [y, 0], [1, x])

    temp_size = tf.cast(
        [tf.math.round(y * size[0]),
         tf.math.round(size[1] - x * size[1]), 3], tf.int32)
    crop_image3, offset = crop_image_and_box(images[2], temp_size)
    box = calculate_boxes_offset(boxes[2],
                                 tf.cast(-offset[0] / size[0], tf.float32),
                                 x - tf.cast(offset[1] / size[1], tf.float32))
    boxes[2], classes[2] = clip_box(box, classes[2], [0, x], [y, 1])

    temp_size = tf.cast([
        tf.math.round(size[0] - y * size[0]),
        tf.math.round(size[1] - x * size[1]), 3
    ], tf.int32)
    crop_image4, offset = crop_image_and_box(images[3], temp_size)
    box = calculate_boxes_offset(boxes[3],
                                 y - tf.cast(offset[0] / size[0], tf.float32),
                                 x - tf.cast(offset[1] / size[1], tf.float32))
    boxes[3], classes[3] = clip_box(box, classes[3], [y, x], [1, 1])

    temp1 = tf.concat([crop_image1, crop_image2], axis=0)
    temp2 = tf.concat([crop_image3, crop_image4], axis=0)
    final = tf.concat([temp1, temp2], axis=1)
    return final, tf.concat(boxes, axis=0), tf.concat(classes, axis=0)

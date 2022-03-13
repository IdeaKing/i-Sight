from typing import Tuple

import tensorflow as tf

from src.utils import anchors as anchors_utils

import numpy as np


def _compute_gt(images, 
                classes, 
                anchors, 
                num_classes):

    labels = classes[0]
    boxes = classes[1]

    target_clf, target_reg = anchors_utils.anchor_targets_bbox(
            anchors, images, boxes, labels, num_classes)

    return target_clf, target_reg



def _generate_anchors(configs,
                      im_shape) -> tf.Tensor:

    anchors_gen = [anchors_utils.AnchorGenerator(
            size=configs.sizes[i - 3],
            aspect_ratios=configs.ratios,
            stride=configs.downsampling_strides[i - 3]) 
            for i in range(3, 8)]

    shapes = [im_shape // (2 ** x) for x in range(3, 8)]

    anchors = [g((size, size, 3))
               for g, size in zip(anchors_gen, shapes)]

    return tf.concat(anchors, axis=0)


def wrap_detection_dataset(ds: tf.data.Dataset,
                           im_size: Tuple[int, int],
                           configs,
                           num_classes: int) -> tf.data.Dataset:

    anchors = _generate_anchors(configs, im_size[0])

    return ds.map(
        lambda x: tf.numpy_function(
            _compute_gt,
            inp=[configs, x, anchors],
            Tout=tf.float32),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
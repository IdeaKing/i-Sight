import numpy as np
import tensorflow as tf

from src.utils.anchors import anchor_targets_bbox, AnchorGenerator


def _compute_gt(images, 
                ground_truth, 
                anchors, 
                num_classes):

    labels = ground_truth[0]
    boxes = ground_truth[1]

    target_clf, target_reg = anchor_targets_bbox(
            anchors, images, boxes, labels, num_classes)

    return target_clf, target_reg


def _generate_anchors(configs,
                      im_shape) -> tf.Tensor:

    anchors_gen = [AnchorGenerator(
            size=configs.sizes[i - 3],
            aspect_ratios=configs.ratios,
            stride=configs.downsampling_strides[i - 3]) 
            for i in range(3, 8)]

    shapes = [im_shape // (2 ** x) for x in range(3, 8)]

    anchors = [g((size, size, 3))
               for g, size in zip(anchors_gen, shapes)]

    return tf.concat(anchors, axis=0)


import numpy as np
import tensorflow as tf

from PIL import Image

from src.utils.bndbox import (regress_bndboxes, clip_boxes, nms)
from src.utils.anchors import (AnchorGenerator)


def _image_to_pil(image):

    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, tf.Tensor):
        image = image.numpy()
    
    if image.dtype == 'float32' or image.dtype == 'float64':
        image = (image * 255.).astype('uint8')
    elif image.dtype != 'uint8':
        print(image.dtype)
        raise ValueError('Image dtype not supported')

    return Image.fromarray(image)


def _parse_box(box):
    if isinstance(box, tf.Tensor):
        return tuple(box.numpy().astype('int32').tolist())
    elif isinstance(box, np.ndarray):
        return tuple(box.astype('int32').tolist())
    else:
        return tuple(map(int, box))


def _parse_boxes(boxes):
    if isinstance(boxes, tf.Tensor):
        boxes = boxes.numpy().astype('int32').tolist()
    elif isinstance(boxes, np.ndarray):
        boxes = boxes.astype('int32').tolist()
    print(boxes)
    return [_parse_box(b) for b in boxes]


class FilterDetections:

    def __init__(self, 
                 anchors_config,
                 score_threshold: float):

        self.score_threshold = score_threshold
        self.anchors_gen = [AnchorGenerator(
            size=anchors_config.sizes[i - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.downsampling_strides[i - 3]
        ) for i in range(3, 8)] # 3 to 7 pyramid levels

        # Accelerate calls
        self.regress_boxes = tf.function(
            regress_bndboxes, input_signature=[
                tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
                tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32)])

        self.clip_boxes = tf.function(
            clip_boxes, input_signature=[
                tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
                tf.TensorSpec(shape=None)])
    
    def __call__(self, 
                 images: tf.Tensor, 
                 regressors: tf.Tensor, 
                 class_scores: tf.Tensor):

        im_shape = tf.shape(images)
        batch_size, h, w = im_shape[0], im_shape[1], im_shape[2]

        # Create the anchors
        shapes = [w // (2 ** x) for x in range(3, 8)]
        anchors = [g((size, size, 3))
                   for g, size in zip(self.anchors_gen, shapes)]
        anchors = tf.concat(anchors, axis=0)
        
        # Tile anchors over batches, so they can be regressed
        anchors = tf.tile(tf.expand_dims(anchors, 0), [batch_size, 1, 1])

        # Regress anchors and clip in case they go outside of the image
        boxes = self.regress_boxes(anchors, regressors)
        boxes = self.clip_boxes(boxes, [h, w])

        # Suppress overlapping detections
        boxes, labels, scores = nms(
            boxes, class_scores, score_threshold=self.score_threshold)

        # TODO: Pad output
        return labels, boxes, scores
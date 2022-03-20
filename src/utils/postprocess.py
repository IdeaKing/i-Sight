import numpy as np
import tensorflow as tf

from typing import Tuple
from PIL import Image

from src.utils import anchors
from src.utils import label_utils


class FilterDetections:

    def __init__(self,
                 score_threshold: float = 0.3,
                 image_shape: Tuple[int, int] = (512, 512),
                 from_logits: bool = False,
                 max_boxes: int = 150,
                 max_size: int = 100,
                 iou_threshold: int = 0.7):

        self.score_threshold = score_threshold
        self.image_shape = image_shape
        self.from_logits = from_logits
        self.max_boxes = max_boxes
        self.max_size = max_size
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold

        self.anchors = anchors.Anchors().get_anchors(
            image_height=image_shape[0],
            image_width=image_shape[1])

    def __call__(self,
                 labels: tf.Tensor,
                 bboxes: tf.Tensor):

        if self.from_logits:
            pred_labels = []
            pred_bboxes = []
            pred_scores = []

            # Loops from each batch
            for label, boxes in zip(labels, bboxes):
                print(label.numpy().shape)
                print(boxes.numpy().shape)
                label = tf.nn.sigmoid(label)
                boxes = label_utils.match_anchors(
                    boxes=boxes,
                    anchor_boxes=self.anchors)
                nms = tf.image.combined_non_max_suppression(
                    tf.expand_dims(boxes, axis=2),
                    label,
                    max_output_size_per_class=self.max_boxes,
                    max_total_size=self.max_size,
                    iou_threshold=self.iou_threshold,
                    score_threshold=self.score_threshold,
                    clip_boxes=False,
                    name="Training-NMS")

                pred_labels.append(nms.nmsed_classes)
                pred_bboxes.append(nms.nmsed_boxes)
                pred_scores.append(nms.nmsed_scores)

            return pred_labels, pred_bboxes, pred_scores

        else:
            labels = tf.nn.sigmoid(labels)
            bboxes = label_utils.match_anchors(
                boxes=bboxes,
                anchor_boxes=self.anchors)
            nms = tf.image.combined_non_max_suppression(
                tf.expand_dims(bboxes, axis=2),
                labels,
                max_output_size_per_class=self.max_boxes,
                max_total_size=self.max_size,
                iou_threshold=self.iou_threshold,
                score_threshold=self.score_threshold,
                clip_boxes=False,
                name="Non-training-NMS")

            labels = nms.nmsed_classes
            bboxes = nms.nmsed_boxes
            scores = nms.nmsed_scores

            return labels, bboxes, scores

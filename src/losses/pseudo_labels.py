import tensorflow as tf
from src.utils import anchors, postprocess

from typing import Tuple

class PseudoLabelObjectDetection():
    """Change the logits into labels and into anchored labels for object detection.
    :params unlabeled_batch_size(int): The unlabeled batch size
    :params image_dims: The size of the input image for the model
    :returns: Pseudo-labels for object detection models
    """

    def __init__(self, 
                 unlabeled_batch_size: int, 
                 image_dims: Tuple[int, int] = (512, 512),
                 score_threshold: float = 0.1,
                 iou_threshold: float = 0.7) -> None:
        self.unlabeled_batch_size = unlabeled_batch_size
        self.image_dims = image_dims
        self.postprocess = postprocess.FilterDetections(
            score_threshold=score_threshold,
            image_dims=self.image_dims,
            iou_threshold=iou_threshold)
        self.anchors = anchors.Anchors()
        self.encoder = anchors.Encoder()
    
    def update_postprocess(self,
                           score_threshold: float,
                           iou_threshold: float):
        self.postprocess = postprocess.FilterDetections(
            score_threshold=score_threshold,
            image_dims=self.image_dims,
            iou_threshold=iou_threshold)

    def __call__(self, logits: Tuple[tf.Tensor, tf.Tensor]):
        pl_images = tf.zeros(
            (self.unlabeled_batch_size, *self.image_dims, 3),
            dtype=tf.float32)

        logits_cls, logits_bbx = logits[0], logits[1]

        # Applies NMS
        pl_cls, pl_bbx, _ = self.postprocess(
            # images=pl_images,
            bboxes=logits_bbx,
            labels=logits_cls)

        # Applies anchors
        _, batched_anchored_cls, batched_anchored_bbx = self.encoder.encode_batch(
            images=pl_images,
            classes=pl_cls,
            gt_boxes=pl_bbx)

        return (batched_anchored_cls, batched_anchored_bbx)


class PseudoLabelClassification():
    None
    # TODO: Write PL Classification
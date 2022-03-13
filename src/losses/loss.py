import numpy as np
import tensorflow as tf

import tensorflow_addons as tfa

from src.utils.training_utils import PseudoLabelObjectDetection


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha: float = 0.25, gamma: float = 1.5) -> None:
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
            alpha=self.alpha, gamma=self.gamma,
            reduction=tf.losses.Reduction.SUM)
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        anchors_states = y_true[:, :, -1]
        labels = y_true[:, :, :-1]

        not_ignore_idx = tf.where(tf.not_equal(anchors_states, -1.))
        true_idx = tf.where(tf.equal(anchors_states, 1.))

        normalizer = tf.shape(true_idx)[0]
        normalizer = tf.cast(normalizer, tf.float32)
        normalizer = tf.maximum(tf.constant(1., dtype=tf.float32), normalizer)

        y_true = tf.gather_nd(labels, not_ignore_idx)
        y_pred = tf.gather_nd(y_pred, not_ignore_idx)

        return tf.divide(self.loss_fn(y_true, y_pred), normalizer)


class HuberLoss(tf.keras.losses.Loss):
    def __init__(self, delta: float = 1.) -> None:
        super(HuberLoss, self).__init__()
        self.delta = delta

        self.loss_fn = tf.losses.Huber(
            reduction=tf.losses.Reduction.SUM, 
            delta=self.delta)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        anchors_states = y_true[:, :, -1]
        labels = y_true[:, :, :-1]

        true_idx = tf.where(tf.equal(anchors_states, 1.))

        normalizer = tf.shape(true_idx)[0]
        normalizer = tf.cast(normalizer, tf.float32)
        normalizer = tf.maximum(tf.constant(1., dtype=tf.float32), normalizer)
        normalizer = tf.multiply(normalizer, tf.constant(4., dtype=tf.float32))

        y_true = tf.gather_nd(labels, true_idx)
        y_pred = tf.gather_nd(y_pred, true_idx)

        return 50. * tf.divide(self.loss_fn(y_true, y_pred), normalizer)


class UDA(tf.keras.losses.Loss):
    """UDA Loss Function."""
    def __init__(self, configs: object = None):
        super(UDA, self).__init__()
        self.configs = configs
        self.convert_to_labels = PseudoLabelObjectDetection(configs)

        self.focal_loss = FocalLoss()
        self.huber_loss = HuberLoss()
        self.consistency_loss = tf.keras.losses.KLDivergence()


    def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor):
        labels = y_true
        masks = {}
        logits = {}
        loss = {}
        # Splits the predictions for labeled, and unlabeled
        logits["l"], logits["u_ori"], logits["u_aug"] = tf.split(
            y_pred,
            [self.configs.batch_size,
                self.configs.unlabeled_batch_size,
                self.configs.unlabeled_batch_size],
            axis = 0)
        # Step 1: Loss for Labeled Values
        labeled_cls_loss = self.focal_loss(labels["l"][0], logits["l"][0])
        labeled_obd_loss = self.huber_loss(labels["l"][1], logits["l"][1])
        loss["l"] = tf.reduce_sum([labeled_cls_loss, labeled_obd_loss])
        # Step 2: Loss for unlabeled values
        labels["u_ori"] = self.convert_to_labels(logits["u_ori"]) # Applies NMS, anchors
        # Consistency loss between unlabeled values
        unlabeled_cls_loss = self.consistency_loss(labels["u_ori"][0], logits["u_aug"][0])
        unlabeled_obd_loss = self.consistency_loss(labels["u_ori"][1], logits["u_aug"][1])
        loss["u"] = tf.reduce_sum([unlabeled_cls_loss, unlabeled_obd_loss])
        return logits, labels, masks, loss

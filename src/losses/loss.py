import numpy as np
import tensorflow as tf

import tensorflow_addons as tfa

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
    def __init__(self, configs):
        super(UDA, self).__init__()
        self.training_type = configs.training_type
        self.configs = configs
        self.convert_to_labels = PseudoLabelObjectDetection(configs)
        self.loss = effdet_loss(configs)
        self.consistency_loss = tf.keras.losses.KLDivergence()

    def __call__(self, y_true, y_pred):
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
        loss["l"] = self.loss(labels["l"], logits["l"])
        # Step 2: Loss for unlabeled values
        labels["u_ori"] = logits["u_ori"] #self.convert_to_labels(logits["u_ori"])
        loss["u"] = self.consistency_loss(labels["u_ori"], logits["u_aug"])
        return logits, labels, masks, loss


class PseudoLabelObjectDetection():
    """Change the logits into labels for object detection.
    :params configs: Configuration class
    :params logits: The outputs from the model
    :returns: Pseudo-labels for object detection models
    """
    def __init__(self, configs):
        self.configs = configs
        self.anchors = Anchors(
            configs=configs)(image_size=configs.image_dims)
        self.nms = NMS(configs=configs)
        self.box_transform = BoxTransform()
        self.clip_boxes = ClipBoxes(configs)
    
    def __call__(self, logits):
        final_out = np.zeros(
            (self.configs.unlabeled_batch_size, 
             self.configs.max_box_num, 
             5),
             dtype=np.float32)
        for i, logit in enumerate(logits):
            reg_results, cls_results = logit[..., :4], logit[..., 4:]
            reg_results = np.expand_dims(reg_results, axis=0)
            cls_results = np.expand_dims(cls_results, axis=0)
            transformed_anchors = self.box_transform(self.anchors, reg_results)
            transformed_anchors = self.clip_boxes(transformed_anchors)
            scores = tf.math.reduce_max(cls_results, axis=2).numpy()
            classes = tf.math.argmax(cls_results, axis=2).numpy()
            final_boxes, final_scores, final_classes = self.nms(
                boxes=transformed_anchors[0, :, :],
                box_scores=np.squeeze(scores),
                box_classes=np.squeeze(classes))
            merged_output = np.concatenate(
                [final_boxes.numpy(), np.expand_dims(final_classes.numpy(),axis=-1)],
                axis=1).tolist()
            num_of_pads = int(self.configs.max_box_num) - int(len(merged_output))
            
            for pad in range(num_of_pads):
                merged_output.append([0, 0, 0, 0, -1])
            final_out[i] = np.array(merged_output)
        return final_out
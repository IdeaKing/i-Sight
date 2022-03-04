# Thomas Chia i-Sight EfficientDet loss functions
# Code from https://github.com/calmisential/EfficientDet_TensorFlow2
# Changes:
#   Separates CLS and OBD Losses

import tensorflow as tf

from src.losses.iou import IOU
from src.losses.anchor import Anchors

def item_assignment(input_tensor, boolean_mask, value, axes):
    """Support item assignment for tf.Tensor
    :param input_tensor: A Tensor
    :param boolean_mask: A Tensor, dtype: tf.bool
    :param value: A scalar
    :param axes : A list of scalar or None, the axes that are not used for masking
    :return: A Tensor with the same dtype as input_tensor
    """
    mask = tf.dtypes.cast(x=boolean_mask, dtype=input_tensor.dtype)
    if axes:
        for axis in axes:
            mask = tf.expand_dims(input=mask, axis=axis)
    masked_tensor = input_tensor * (1 - mask)
    masked_value = value * mask
    assigned_tensor = masked_tensor + masked_value
    return assigned_tensor


def advanced_item_assignmnet(input_tensor, boolean_mask, value, target_elements, elements_axis):
    """Supports assignment of specific elements for tf.Tensor
    :param input_tensor: A Tensor
    :param boolean_mask: A Tensor, dtype: tf.bool
    :param value: A scalar
    :param target_elements: A Tensor, shape: (N,), which specifies the index of the element to be assigned.
    :param elements_axis: A scalar, the axis of specific elements
    :return:
    """
    target_elements = item_assignment(target_elements, ~boolean_mask, -1, None)
    mask = tf.one_hot(indices=tf.cast(target_elements, dtype=tf.int32),
                      depth=input_tensor.shape[elements_axis],
                      axis=-1,
                      dtype=tf.float32)
    assigned_tensor = input_tensor * (1 - mask) + value * mask
    return assigned_tensor


class FocalLoss:
    def __init__(self, configs):
        """ Calculates the L1 and Regression Loss for Object Detection.
        :params configs: Class with configurations
        :returns: A single loss value
        """
        self.configs = configs
        self.alpha = configs.alpha
        self.gamma = configs.gamma
        self.anchors = Anchors(
            scales = configs.scales, 
            ratios = configs.ratios,
            configs = configs)(image_size = self.configs.image_dims)

    def __call__(self, y_true, y_pred):
        reg_results, cls_results = y_pred[..., :4], y_pred[..., 4:]
        assert cls_results.shape[0] == reg_results.shape[0]
        batch_size = cls_results.shape[0]
        cls_loss_list = []
        reg_loss_list = []

        labels = tf.cast(y_true, dtype=tf.float32).numpy()
        anchor = self.anchors[0, :, :]

        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_center_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_center_y = anchor[:, 1] + 0.5 * anchor_heights

        for n in range(batch_size):
            class_result = cls_results[n, :, :]
            reg_result = reg_results[n, :, :]

            box_annotation = labels[n, :, :]
            # Filter out the extra padding boxes.
            box_annotation = box_annotation[box_annotation[:, 4] != -1]

            if box_annotation.shape[0] == 0:
                cls_loss_list.append(
                    tf.cast(tf.reduce_sum(cls_results)*0, tf.float32))
                reg_loss_list.append(
                    tf.cast(tf.reduce_sum(reg_results)*0, tf.float32))
                continue

            class_result = tf.clip_by_value(
                t=class_result, clip_value_min=1e-4, clip_value_max=1.0 - 1e-4)

            iou_value = IOU(box_1=anchor, box_2=box_annotation[:, :4]).calculate_iou()
            iou_max = tf.math.reduce_max(iou_value, axis=1)
            iou_argmax = tf.math.argmax(iou_value, axis=1)
            targets = tf.ones_like(class_result) * -1
            targets = item_assignment(
                input_tensor=targets,
                boolean_mask=tf.math.less(iou_max, 0.4),
                value=0,
                axes=[1])
            positive_indices = tf.math.greater(iou_max, 0.5)
            num_positive_anchors = tf.reduce_sum(
                tf.dtypes.cast(x=positive_indices, dtype=tf.int32))
            
            assigned_annotations = box_annotation[iou_argmax, :]

            targets = item_assignment(
                input_tensor=targets,
                boolean_mask=positive_indices,
                value=0,
                axes=[1])

            targets = advanced_item_assignmnet(
                input_tensor=targets,
                boolean_mask=positive_indices,
                value=1,
                target_elements=tf.convert_to_tensor(
                    assigned_annotations[:, 4], dtype=tf.float32),
                elements_axis=1)
            
            alpha_factor = tf.ones_like(targets) * self.alpha
            alpha_factor = tf.where(tf.math.equal(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = tf.where(tf.math.equal(targets, 1.), 1. - class_result, class_result)
            focal_weight = alpha_factor * tf.math.pow(focal_weight, self.gamma)
            bce = -(targets * tf.math.log(class_result) + (1.0 - targets) * tf.math.log(1.0 - class_result))

            cls_loss = focal_weight * bce
            cls_loss = tf.where(
                tf.math.not_equal(targets, -1.0), 
                cls_loss, 
                tf.zeros_like(cls_loss))
            cls_loss_list.append(
                tf.math.reduce_sum(cls_loss) / tf.keras.backend.clip(
                    x=tf.cast(num_positive_anchors, dtype=tf.float32), 
                    min_value=1.0, 
                    max_value=None))

            if num_positive_anchors > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_center_x_pi = anchor_center_x[positive_indices]
                anchor_center_y_pi = anchor_center_y[positive_indices]
                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_center_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_center_y = assigned_annotations[:, 1] + 0.5 * gt_heights
                gt_widths = tf.keras.backend.clip(x=gt_widths, min_value=1, max_value=None)
                gt_heights = tf.keras.backend.clip(x=gt_heights, min_value=1, max_value=None)

                targets_dx = (gt_center_x - anchor_center_x_pi) / anchor_widths_pi
                targets_dy = (gt_center_y - anchor_center_y_pi) / anchor_heights_pi
                targets_dw = tf.math.log(gt_widths / anchor_widths_pi)
                targets_dh = tf.math.log(gt_heights / anchor_heights_pi)
                targets = tf.stack([targets_dx, targets_dy, targets_dw, targets_dh])
                targets = tf.transpose(a=targets, perm=[1, 0])
                targets = targets / tf.constant([[0.1, 0.1, 0.2, 0.2]])

                reg_diff = tf.math.abs(
                    targets - tf.boolean_mask(reg_result, positive_indices, axis=0))
                reg_loss = tf.where(
                    tf.math.less_equal(reg_diff, 1.0 / 9.0), 
                    0.5 * 9.0 * tf.math.pow(reg_diff, 2), 
                    reg_diff - 0.5 / 9.0)
                reg_loss_list.append(tf.reduce_mean(reg_loss))
            else:
                reg_loss_list.append(
                    tf.cast(tf.reduce_sum(reg_results)*0, tf.float32))
        final_cls_loss = tf.math.reduce_mean(
            tf.stack(cls_loss_list, axis=0), axis=0, keepdims=False)
        # print("final_cls", final_cls_loss)
        final_reg_loss = tf.math.reduce_mean(
            tf.stack(reg_loss_list, axis=0), axis=0, keepdims=False)
        # print("final reg", final_reg_loss)
        loss = tf.math.reduce_sum([final_cls_loss, final_reg_loss])

        return loss


def obdloss(y_true, y_pred):
    """Computes the total loss for object detection."""
    reg_results, cls_results = y_pred[..., :4], y_pred[..., 4:]
    reg_true, cls_true = y_pred[..., :4], y_pred[..., 4:]



class OfficialFocalLoss(tf.keras.losses.Loss):
  """Compute the focal loss between `logits` and the golden `target` values.
  Focal loss = -(1-pt)^gamma * log(pt)
  where pt is the probability of being classified to the true class.
  """

  def __init__(self, alpha, gamma, label_smoothing=0.0, **kwargs):
    """Initialize focal loss.
    Args:
      alpha: A float32 scalar multiplying alpha to the loss from positive
        examples and (1-alpha) to the loss from negative examples.
      gamma: A float32 scalar modulating loss from hard and easy examples.
      label_smoothing: Float in [0, 1]. If > `0` then smooth the labels.
      **kwargs: other params.
    """
    super().__init__(**kwargs)
    self.alpha = alpha
    self.gamma = gamma
    self.label_smoothing = label_smoothing

  @tf.autograph.experimental.do_not_convert
  def call(self, y, y_pred):
    """Compute focal loss for y and y_pred.
    Args:
      y: A tuple of (normalizer, y_true), where y_true is the target class.
      y_pred: A float32 tensor [batch, height_in, width_in, num_predictions].
    Returns:
      the focal loss.
    """
    normalizer, y_true = y
    alpha = tf.convert_to_tensor(self.alpha, dtype=y_pred.dtype)
    gamma = tf.convert_to_tensor(self.gamma, dtype=y_pred.dtype)

    # compute focal loss multipliers before label smoothing, such that it will
    # not blow up the loss.
    pred_prob = tf.sigmoid(y_pred)
    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = (1.0 - p_t)**gamma

    # apply label smoothing for cross_entropy for each entry.
    y_true = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    # compute the final loss and return
    return alpha_factor * modulating_factor * ce / normalizer


class BoxLoss(tf.keras.losses.Loss):
  """L2 box regression loss."""

  def __init__(self, delta=0.1, **kwargs):
    """Initialize box loss.
    Args:
      delta: `float`, the point where the huber loss function changes from a
        quadratic to linear. It is typically around the mean value of regression
        target. For instances, the regression targets of 512x512 input with 6
        anchors on P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
      **kwargs: other params.
    """
    super().__init__(**kwargs)
    self.huber = tf.keras.losses.Huber(
        delta, reduction=tf.keras.losses.Reduction.NONE)

  @tf.autograph.experimental.do_not_convert
  def call(self, y_true, box_outputs):
    num_positives, box_targets = y_true
    normalizer = num_positives * 4.0
    mask = tf.cast(box_targets != 0.0, box_outputs.dtype)
    box_targets = tf.expand_dims(box_targets, axis=-1)
    box_outputs = tf.expand_dims(box_outputs, axis=-1)
    # TODO(fsx950223): remove cast when huber loss dtype is fixed.
    box_loss = tf.cast(self.huber(box_targets, box_outputs),
                       box_outputs.dtype) * mask
    box_loss = tf.reduce_sum(box_loss) / normalizer
    return box_loss
# Thomas Chia i-Sight EfficientDet loss functions
# Code from https://github.com/calmisential/EfficientDet_TensorFlow2
# Changes:
#   Separates CLS and OBD Losses

import tensorflow as tf

from src.losses.iou import IOU
from src.losses.anchor import Anchors


def item_assignment(input_tensor, boolean_mask, value, axes):
    """
    Support item assignment for tf.Tensor
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
    """
    Supports assignment of specific elements for tf.Tensor
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
    """Classificiation Loss."""
    def __init__(self, configs=None):
        self.configs = configs
        self.anchors = Anchors(
            scales = configs.scales, 
            ratios = configs.ratios,
            configs = configs)(image_size = self.configs.image_dims)


    def __call__(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        box_logits, cls_logits = y_pred[..., :4], y_pred[..., 4:]
        batch_size = cls_logits.shape[0]
        loss_values = []
        # Define the Anchors Locations
        anchor = self.anchors[0, :, :]
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        # Parse through each example
        for n in range(batch_size):
            class_result = cls_logits[n, :, :]
            box_annotation = y_true[n, :, :]
            # Filter out the extra padding boxes.
            box_annotation = box_annotation[box_annotation[:, 4] != -1]
            if box_annotation.shape[0] == 0:
                loss_values.append(tf.constant(0, dtype=tf.dtypes.float32))
                continue
            class_result = tf.clip_by_value(
                t=class_result, 
                clip_value_min=1e-4, 
                clip_value_max=1.0 - 1e-4)
            iou_value = IOU(
                box_1=anchor, 
                box_2=box_annotation[:, :4]).calculate_iou()
            iou_max = tf.math.reduce_max(iou_value, axis=1)
            iou_argmax = tf.stop_gradient(tf.math.argmax(iou_value, axis=1))
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
                    assigned_annotations[:, 4], 
                    dtype=tf.float32),
                elements_axis=1)

            alpha_factor = tf.ones_like(targets) * self.configs.alpha
            alpha_factor = tf.where(
                tf.math.equal(targets, 1.), 
                alpha_factor, 
                1. - alpha_factor)
            focal_weight = tf.where(
                tf.math.equal(targets, 1.), 
                1. - class_result, 
                class_result)
            focal_weight = alpha_factor * tf.math.pow(
                focal_weight, self.configs.gamma)
            bce = -(targets * tf.math.log(class_result) + \
                (1.0 - targets) * tf.math.log(1.0 - class_result))

            cls_loss = focal_weight * bce
            cls_loss = tf.where(
                tf.math.not_equal(targets, -1.0), 
                cls_loss, 
                tf.zeros_like(cls_loss))
            loss_values.append(
                tf.math.reduce_sum(cls_loss) / \
                tf.keras.backend.clip(
                    x=tf.cast(num_positive_anchors, 
                    dtype=tf.float32), 
                min_value=1.0, 
                max_value=None))
            
            return tf.math.reduce_mean(
                tf.stack(loss_values, axis=0), 
                axis=0, 
                keepdims=True)


class RegressionLoss:
    """Localization Loss."""
    def __init__(self, configs=None):
        self.configs = configs
        self.anchors = Anchors(
            scales = configs.scales, 
            ratios = configs.ratios,
            configs = configs)(image_size = self.configs.image_dims)

    
    def __call__(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        box_logits, cls_logits = y_pred[..., :4], y_pred[..., 4:]
        batch_size = box_logits.shape[0]
        loss_values = []
        # Define the Anchors Locations
        anchor = self.anchors[0, :, :]
        anchor_widths = anchor[:, 2] - anchor[:, 0]
        anchor_heights = anchor[:, 3] - anchor[:, 1]
        anchor_center_x = anchor[:, 0] + 0.5 * anchor_widths
        anchor_center_y = anchor[:, 1] + 0.5 * anchor_heights
        # Parse through each example
        for n in range(batch_size):
            class_result = cls_logits[n, :, :]
            reg_result = box_logits[n, :, :]
            box_annotation = y_true[n, :, :]
            # Filter out the extra padding boxes.
            box_annotation = box_annotation[box_annotation[:, 4] != -1]
            if box_annotation.shape[0] == 0:
                loss_values.append(tf.constant(0, dtype=tf.dtypes.float32))
                continue
            class_result = tf.clip_by_value(
                t=class_result, 
                clip_value_min=1e-4, 
                clip_value_max=1.0 - 1e-4)
            iou_value = IOU(
                box_1=anchor, 
                box_2=box_annotation[:, :4]).calculate_iou()
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
                gt_widths = tf.keras.backend.clip(
                    x=gt_widths, min_value=1, max_value=None)
                gt_heights = tf.keras.backend.clip(
                    x=gt_heights, min_value=1, max_value=None)

                targets_dx = (gt_center_x - anchor_center_x_pi) / anchor_widths_pi
                targets_dy = (gt_center_y - anchor_center_y_pi) / anchor_heights_pi
                targets_dw = tf.math.log(
                    gt_widths / anchor_widths_pi)
                targets_dh = tf.math.log(
                    gt_heights / anchor_heights_pi)
                targets = tf.stack(
                    [targets_dx, targets_dy, targets_dw, targets_dh])
                targets = tf.transpose(
                    a=targets, perm=[1, 0])
                targets = targets / tf.constant([[0.1, 0.1, 0.2, 0.2]])

                reg_diff = tf.math.abs(
                    targets - tf.boolean_mask(reg_result, positive_indices, axis=0))
                reg_loss = tf.where(
                    tf.math.less_equal(
                        reg_diff, 1.0 / 9.0), 0.5 * 9.0 * tf.math.pow(reg_diff, 2), 
                        reg_diff - 0.5 / 9.0)
                loss_values.append(tf.reduce_mean(reg_loss))
            else:
                loss_values.append(tf.constant(0, dtype=tf.float32))
        
        return tf.math.reduce_mean(
            tf.stack(loss_values, axis=0), 
            axis=0, 
            keepdims=True)
import numpy as np
import tensorflow as tf

from src.models.efficientdet import BoxTransform, ClipBoxes, MapToInputImage
from src.losses.anchor import Anchors
from src.losses.iou import IOU
from src.utils.tools import item_assignment, advanced_item_assignmnet
from src.utils.nms import NMS


def effdet_loss(configs):
    anchors = Anchors(configs=configs)(image_size=configs.image_dims)
    loss = FocalLoss(configs=configs)

    def training_procedure(y_true, y_pred):
        reg_results, cls_results = y_pred[..., :4], y_pred[..., 4:]

        cls_loss_value, reg_loss_value = loss(cls_results, reg_results, anchors, y_true)
        loss_value = tf.math.reduce_mean(cls_loss_value) + tf.reduce_mean(reg_loss_value)
        return loss_value
    return training_procedure


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
             5))
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
        return tf.constant(final_out, dtype=tf.float32)


class FocalLoss:
    def __init__(self, configs):
        self.alpha = configs.alpha
        self.gamma = configs.gamma

    def __call__(self, cls_results, reg_results, anchors, labels):
        assert cls_results.shape[0] == reg_results.shape[0]
        batch_size = cls_results.shape[0]
        cls_loss_list = []
        reg_loss_list = []

        anchor = anchors[0, :, :]

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
                cls_loss_list.append(tf.constant(0, dtype=tf.dtypes.float32))
                reg_loss_list.append(tf.constant(0, dtype=tf.dtypes.float32))
                continue

            class_result = tf.clip_by_value(t=class_result, clip_value_min=1e-4, clip_value_max=1.0 - 1e-4)

            iou_value = IOU(box_1=anchor, box_2=box_annotation[:, :4]).calculate_iou()
            iou_max = tf.math.reduce_max(iou_value, axis=1)
            iou_argmax = tf.math.argmax(iou_value, axis=1)

            targets = tf.ones_like(class_result) * -1
            targets = item_assignment(input_tensor=targets,
                                      boolean_mask=tf.math.less(iou_max, 0.4),
                                      value=0,
                                      axes=[1])

            positive_indices = tf.math.greater(iou_max, 0.5)
            num_positive_anchors = tf.reduce_sum(tf.dtypes.cast(x=positive_indices, dtype=tf.int32))
            assigned_annotations = box_annotation[iou_argmax, :]

            targets = item_assignment(input_tensor=targets,
                                      boolean_mask=positive_indices,
                                      value=0,
                                      axes=[1])

            targets = advanced_item_assignmnet(input_tensor=targets,
                                               boolean_mask=positive_indices,
                                               value=1,
                                               target_elements=tf.convert_to_tensor(assigned_annotations[:, 4], dtype=tf.float32),
                                               elements_axis=1)

            alpha_factor = tf.ones_like(targets) * self.alpha
            alpha_factor = tf.where(tf.math.equal(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = tf.where(tf.math.equal(targets, 1.), 1. - class_result, class_result)
            focal_weight = alpha_factor * tf.math.pow(focal_weight, self.gamma)
            bce = -(targets * tf.math.log(class_result) + (1.0 - targets) * tf.math.log(1.0 - class_result))

            cls_loss = focal_weight * bce
            cls_loss = tf.where(tf.math.not_equal(targets, -1.0), cls_loss, tf.zeros_like(cls_loss))
            cls_loss_list.append(tf.math.reduce_sum(cls_loss) / tf.keras.backend.clip(x=tf.cast(num_positive_anchors, dtype=tf.float32), min_value=1.0, max_value=None))

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

                reg_diff = tf.math.abs(targets - tf.boolean_mask(reg_result, positive_indices, axis=0))
                reg_loss = tf.where(tf.math.less_equal(reg_diff, 1.0 / 9.0), 0.5 * 9.0 * tf.math.pow(reg_diff, 2), reg_diff - 0.5 / 9.0)
                reg_loss_list.append(tf.reduce_mean(reg_loss))
            else:
                reg_loss_list.append(tf.constant(0, dtype=tf.float32))

        final_cls_loss = tf.math.reduce_mean(tf.stack(cls_loss_list, axis=0), axis=0, keepdims=True)
        final_reg_loss = tf.math.reduce_mean(tf.stack(reg_loss_list, axis=0), axis=0, keepdims=True)

        return final_cls_loss, final_reg_loss


class UDA:
    """UDA Loss Function."""
    def __init__(self, configs):
        self.training_type = configs.training_type
        self.configs = configs
        self.convert_to_labels = PseudoLabelObjectDetection(configs)
        self.loss = FocalLoss(configs)
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
        labels["u_ori"] = self.convert_to_labels(logits["u_ori"])
        loss["u"] = self.consistency_loss(labels["u_ori"], logits["u_aug"])
        return logits, labels, masks, loss
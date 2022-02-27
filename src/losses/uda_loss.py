# Thomas Chia i-Sight UDA Losses

import numpy as np
import tensorflow as tf

import src.losses.effdet_loss as effdet_loss

class UDA:
    """UDA Loss Function."""
    def __init__(self, configs):
        self.training_type = configs.training_type
        self.configs = configs
        if self.training_type == "obd":
            self.loss = effdet_loss.FocalLoss(configs)
            self.consistency_loss = tf.keras.losses.KLDivergence()

    def __call__(self, y_true, y_pred):
        if self.training_type == "obd":
            labels = y_true #{"l": y_true}
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
            loss["l"] = self.loss(labels["l"], logits["l"])#cls_results, reg_results, labels["l"])
            loss["l"] = loss["l"] / float(self.configs.batch_size)
            # Step 2: Loss for unlabeled values
            labels["u_ori"] = logits["u_ori"]
            loss["u"] = self.consistency_loss(labels["u_ori"], logits["u_aug"])
            return logits, labels, masks, loss

"""
class UDA:
    # ""UDA Loss Function.""
    def __init__(self, configs):
        self.training_type = configs.training_type
        self.configs = configs

        if self.training_type == "obd":
            self.cls_loss = effdet_loss.FocalLoss(
                configs = configs)
            self.box_loss = effdet_loss.RegressionLoss(
                configs = configs)

    def __call__(self, y_true, y_pred):
        if self.training_type == "obd":
            labels = {"l": y_true}
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
            loss["l"] = tf.math.reduce_mean(self.cls_loss(labels["l"], logits["l"])) + \
                        tf.math.reduce_mean(self.box_loss(labels["l"], logits["l"]))
            loss["l"] = tf.reduce_sum(loss["l"] / float(self.configs.batch_size))

            # Step 2: Loss for unlabeled values
            labels["u_ori"] = tf.nn.sigmoid(
                logits["u_ori"] / tf.convert_to_tensor(self.configs.uda_label_temperature))
            labels["u_ori"] = tf.stop_gradient(labels["u_ori"])

            loss["u"] = (labels["u_ori"] * tf.nn.log_softmax(logits["u_aug"], axis = -1))

            largest_probs = tf.reduce_max(
                labels["u_ori"], axis = -1, keepdims = True)
            masks["u"] = tf.math.greater_equal(
                largest_probs, 
                tf.constant(self.configs.uda_threshold))
            masks["u"] = tf.cast(masks["u"], tf.float32)
            masks["u"] = tf.stop_gradient(masks["u"])
            loss["u"] = tf.reduce_sum(-loss["u"] * masks["u"]) / \
                        tf.convert_to_tensor(
                            self.configs.unlabeled_batch_size, 
                            dtype = tf.float32)

            return logits, labels, masks, loss"""

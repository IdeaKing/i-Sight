import tensorflow as tf
import numpy as np

from src.models.efficientnet import EfficientNet
from src.models.bifpn import BiFPN
from src.models.prediction_net import BoxClassPredict


def model_builder(configs, name):
    """Creates the EfficienDet model."""
    model_input = tf.keras.layers.Input(
        (configs.image_dims[0], 
         configs.image_dims[1], 
         3))
    backbone = EfficientNet(
        width_coefficient = configs.width_coefficient,
        depth_coefficient = configs.depth_coefficient,
        dropout_rate = configs.dropout_rate)(model_input)
    bifpn_det = BiFPN(
        output_channels = configs.w_bifpn,
        layers = configs.d_bifpn)(backbone)
    prediction_net = BoxClassPredict(
        filters = configs.w_bifpn,
        depth = configs.d_class,
        num_classes = configs.num_classes,
        num_anchors = configs.anchors)(bifpn_det)
    model = tf.keras.models.Model(
        inputs = model_input, 
        outputs = prediction_net,
        name = "EfficientDet-" + \
            str(configs.network_type) + \
            "-" + str(name))
    # model.summary()
    return model


class BoxTransform:
    def __call__(self, boxes, deltas, *args, **kwargs):
        deltas = deltas #.numpy()
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        center_x = boxes[:, :, 0] + 0.5 * widths
        center_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * 0.1
        dy = deltas[:, :, 1] * 0.1
        dw = deltas[:, :, 2] * 0.2
        dh = deltas[:, :, 3] * 0.2

        pred_center_x = center_x + dx * widths
        pred_center_y = center_y + dy * heights
        pred_w = np.exp(dw) * widths
        pred_h = np.exp(dh) * heights

        pred_boxes_x1 = pred_center_x - 0.5 * pred_w
        pred_boxes_y1 = pred_center_y - 0.5 * pred_h
        pred_boxes_x2 = pred_center_x + 0.5 * pred_w
        pred_boxes_y2 = pred_center_y + 0.5 * pred_h

        pred_boxes = np.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], axis=2)

        return pred_boxes


class ClipBoxes:
    def __init__(self, configs):
        self.height, self.width = configs.image_dims[0], configs.image_dims[1]

    def __call__(self, boxes, *args, **kwargs):
        boxes[:, :, 0] = np.clip(a=boxes[:, :, 0], a_min=0, a_max=self.width - 1)
        boxes[:, :, 1] = np.clip(a=boxes[:, :, 1], a_min=0, a_max=self.height - 1)
        boxes[:, :, 2] = np.clip(a=boxes[:, :, 2], a_min=0, a_max=self.width - 1)
        boxes[:, :, 3] = np.clip(a=boxes[:, :, 3], a_min=0, a_max=self.height - 1)
        return boxes


class MapToInputImage:
    def __init__(self, input_image_size, configs):
        self.h, self.w = input_image_size
        self.x_ratio = self.w / configs.image_dims[0]
        self.y_ratio = self.h / configs.image_dims[1]

    def __call__(self, boxes, *args, **kwargs):
        boxes[:, :, 0] = boxes[:, :, 0] * self.x_ratio
        boxes[:, :, 1] = boxes[:, :, 1] * self.y_ratio
        boxes[:, :, 2] = boxes[:, :, 2] * self.x_ratio
        boxes[:, :, 3] = boxes[:, :, 3] * self.y_ratio
        return boxes


"""
class PostProcessing:
    def __init__(self, configs):
        self.anchors = Anchors(
            scales=[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], 
            ratios=[0.5, 1, 2])
        self.loss = FocalLoss()
        self.image_size = (
                configs.image_dims, 
                configs.image_dims)

    def training_procedure(self, efficientdet_ouputs, labels):
        anchors = self.anchors(
            image_size=self.image_size)
        reg_results, cls_results = efficientdet_ouputs[..., :4], efficientdet_ouputs[..., 4:]
        cls_loss_value, reg_loss_value = self.loss(cls_results, reg_results, anchors, labels)
        loss_value = tf.math.reduce_mean(cls_loss_value) + tf.reduce_mean(reg_loss_value)
        return loss_value

    def testing_procedure(self, efficientdet_ouputs, input_image_size):
        box_transform = BoxTransform()
        clip_boxes = ClipBoxes()
        map_to_original = MapToInputImage(input_image_size)
        nms = NMS()

        anchors = self.anchors(image_size=self.image_size)
        reg_results, cls_results = efficientdet_ouputs[..., :4], efficientdet_ouputs[..., 4:]

        transformed_anchors = box_transform(anchors, reg_results)
        transformed_anchors = clip_boxes(transformed_anchors)
        transformed_anchors = map_to_original(transformed_anchors)
        scores = tf.math.reduce_max(cls_results, axis=2).numpy()
        classes = tf.math.argmax(cls_results, axis=2).numpy()
        final_boxes, final_scores, final_classes = nms(
            boxes=transformed_anchors[0, :, :],
            box_scores=np.squeeze(scores),
            box_classes=np.squeeze(classes))
        return final_boxes.numpy(), final_scores.numpy(), final_classes.numpy()
"""
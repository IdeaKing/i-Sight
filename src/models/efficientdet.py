import tensorflow as tf

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
    return model

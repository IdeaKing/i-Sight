import tensorflow as tf

from .backbones import get_backbone
from .layers import Interp, PyramidPooling

class PSPNet(tf.keras.models.Model):
    def __init__(self, 
                 input_dims: tuple = (512, 512),
                 backbone: str = "efficientnet_b0",
                 num_classes: int = 3,
                 dropout_rate: float = 0.2,
                 from_pretrained: bool = False):
        """Builds PSP Net on EffcientNet Backbone."""
        super(PSPNet, self).__init__()
        self.input_dims = input_dims
        self.num_classes = num_classes
        self.backbone = get_backbone(backbone)
        self.dropout_rate = dropout_rate
        if from_pretrained:
            self.backbone.trainable = False
        else:
            self.backbone.trainable = True
        
        # Creates the kernel strides map
        if self.input_dims == (473, 473):
            self.kernel_strides_map = {1: 60,
                                       2: 30,
                                       3: 20,
                                       6: 10}
        elif self.input_dims == (713, 713):
            self.kernel_strides_map = {1: 90,
                                       2: 45,
                                       3: 30,
                                       6: 15}
        elif self.input_dims == (512, 512):
            self.kernel_strides_map = {1: 70,
                                       2: 40,
                                       3: 25,
                                       6: 10}
        else:
            print("Pooling parameters for input shape ",
                self.input_dims, " are not defined.")
            exit(1)
        
        # Segmentation models
        self.pyramid_pooling_module = PyramidPooling(
            kernel_strides_map=self.kernel_strides_map,
            input_dims=self.input_dims)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=512, 
            kernel_size=(3, 3), 
            strides=(1, 1), 
            padding="same", 
            name="conv5_4",
            use_bias=False)
        self.bn1 = tf.keras.layers.BatchNormalization(name="conv5_4_bn")
        self.act1 = tf.keras.layers.Activation("relu")
        self.drop1 = tf.keras.layers.Dropout(0.1)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=self.num_classes,
            kernel_size=(1, 1),
            strides=(1, 1),
            name="conv6")
        self.interp = Interp([*self.input_dims])
        self.act2 = tf.keras.layers.Activation("softmax")

    def call(self, input, training=False):
        x = self.backbone(input, training=training)[2] # We only want the final out of the model
        x = self.pyramid_pooling_module(x, training=training)
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        x = self.drop1(x, training=training)
        x = self.conv2(x)
        x = self.interp(x)
        x = self.act2(x)
        return x



def get_pspnet(name: str = "pspnet_s0",
               input_dims: tuple = (512, 512),
               num_classes: int = 3,
               from_pretrained: bool = False) -> tf.keras.models.Model:
    """Builds PSP Net on EffcientNet Backbone."""

    models = {
        "pspnet_s0": (0.1, "efficientnet_b0"),
        "pspnet_s1": (0.1, "efficientnet_b1"),
        "pspnet_s2": (0.1, "efficientnet_b2"),
        "pspnet_s3": (0.15, "efficientnet_b3"),
        "pspnet_s4": (0.15, "efficientnet_b4"),
        "pspnet_s5": (0.2, "efficientnet_b5"),
        "pspnet_s6": (0.2, "efficientnet_b6"),
        "pspnet_s7": (0.2, "efficientnet_b7"),
    }
    input_layer = tf.keras.layers.Input((*input_dims, 3))
    psp_net = PSPNet(input_dims=input_dims,
                     backbone=models[name][1],
                     num_classes=num_classes,
                     dropout_rate=models[name][0],
                     from_pretrained=from_pretrained)(input_layer)
    return tf.keras.models.Model(inputs=[input_layer], 
                                 outputs=[psp_net])
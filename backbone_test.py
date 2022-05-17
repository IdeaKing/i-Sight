import tensorflow as tf

from src.models import backbones

if __name__ == "__main__":
    bb = tf.keras.applications.EfficientNetB0(include_top=False,
                            weights="imagenet",
                            input_shape=[512, 512, 3])
    model = tf.keras.Model(inputs=bb.inputs, outputs=bb.get_layer("top_activation").output)

    model.summary()
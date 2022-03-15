import tensorflow as tf
from src.models.backbones import backbone_factory
from src.models.backbones import efficientnet_builder

if __name__=="__main__":
    images = tf.zeros((1, 224, 224, 3), dtype=tf.float32)
    efficientnet_builder.build_model(
        images,
        model_name="efficientnet-b0",
        override_params=None,
        training=False,
        features_only=False,
        pooled_features_only=False)
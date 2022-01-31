import tensorflow as tf
import src.models.efficientdet as efficientdet
import configuration as Configs

Config = Configs.Config

if __name__ == "__main__":
    sample_inputs = tf.random.normal(shape=(Config.batch_size, Config.get_image_size()[0], Config.get_image_size()[1], Config.image_channels))
    net = efficientdet.EfficientDet()
    sample_outputs = net(sample_inputs, training=True)
    net.summary()    
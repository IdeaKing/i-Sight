# Test the UDA loss file

import tensorflow as tf
import numpy as np

import src.losses.uda_loss as uda_loss
import src.models.efficientdet as efficientdet

class configs:
    batch_size = 1
    unlabeled_batch_size = 2
    uda_label_temperature = 0.7
    uda_threshold = 0.5
    max_box_num = 200
    anchors = 9
    ratios = [0.5, 1, 2]
    scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    downsampling_strides = [8, 16, 32, 64, 128]
    sizes = [32, 64, 128, 256, 512]
    image_dims = (512, 512)
    width_coefficient = 1.0
    depth_coefficient = 1.0
    dropout_rate = 0.2
    w_bifpn = 64
    d_bifpn = 2
    d_class = 3
    alpha = 0.25
    gamma = 2.00
    num_classes = 8
    network_type = "d0"

if __name__ == "__main__":
    uda_func = uda_loss.UDA(
        training_type = "obd", 
        configs = configs)

    l_images = tf.convert_to_tensor(
        np.random.random((
            configs.batch_size, 512, 512, 3)),
        tf.float32)
    u_orgim = tf.convert_to_tensor(
        np.random.random((
            configs.unlabeled_batch_size, 512, 512, 3)),
        tf.float32)
    u_augim = tf.convert_to_tensor(
        np.random.random((
            configs.unlabeled_batch_size, 512, 512, 3)),
        tf.float32)
    all_images = tf.concat(
        [l_images, u_orgim, u_augim], 
        axis = 0)
    
    l_labels = np.array(
        [[[25, 100, 50, 150, 1], [20, 100, 50, 150, 1]],
         [[25, 100, 50, 150, 1], [20, 100, 50, 150, 1]]], 
        np.float32)

    model = efficientdet.model_builder(
        configs = configs, 
        name = "test")

    logits = model(all_images)

    _, _, _, loss = uda_func(l_labels, logits)

    print(loss)
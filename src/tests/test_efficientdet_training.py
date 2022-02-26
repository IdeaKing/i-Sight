# Tests basic training for EfficientDet

import tensorflow as tf

import src.dataset as dataset
import src.config as config
import src.losses.effdet_loss as effdet_loss
import src.models.efficientnet as efficientnet
import src.models.bifpn as bifpn
import src.models.prediction_net as prednet

def model_builder(configs):
    """Creates the EfficienDet model."""
    backbone = efficientnet.EfficientNet(
        width_coefficient = configs.width_coefficient,
        depth_coefficient = configs.depth_coefficient,
        dropout_rate = configs.dropout_rate)
    bifpn_det = bifpn.BiFPN(
        output_channels = configs.w_bifpn,
        layers = configs.d_bifpn)
    prediction_net = prednet.BoxClassPredict(
        filters = configs.w_bifpn,
        depth = configs.d_class,
        num_classes = configs.num_classes,
        num_anchors = configs.anchors)
    model_input = tf.keras.layers.Input(
        (configs.image_dims[0], 
         configs.image_dims[1], 
         3))
    x = backbone(model_input)
    x = bifpn_det(x)
    x = prediction_net(x)
    model = tf.keras.models.Model(
        inputs = model_input, 
        outputs = x,
        name = "EfficientDet-" + configs.network_type)
    model.summary()
    
    return model



def train(dataset, configs):
    """Training loop for EfficientDet."""
    model = model_builder(configs)
    optimizer = tf.keras.optimizers.Adam(lr=0.001)
    focal_loss = effdet_loss.FocalLoss(configs = configs)
    regression_loss = effdet_loss.RegressionLoss(configs = configs)
    loss_metric = tf.metrics.Mean()
    step = 0
    for epoch in range(configs.epochs):
        for l_image, l_label in dataset:
            with tf.GradientTape() as tape:
                logits = model(l_image, training = True)
                cls_loss = focal_loss(
                    y_true = l_label.numpy(),
                    y_pred = logits)
                box_loss = regression_loss(
                    y_true = l_label.numpy(),
                    y_pred = logits)
                loss = tf.math.reduce_mean(cls_loss) + \
                    tf.math.reduce_mean(box_loss)
            gradients = tape.gradient(
                loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            print("Step: ", step," Loss: ", loss)

def main():
    configs = config.Configs(
        training_type = "obd",
        dataset_path = "datasets/obd_fundus",
        training_dir = "training_dir/test-001")
    image_files = dataset.get_files(configs=configs)
    training_dataset = dataset.Dataset(
        file_names = image_files,
        configs = configs).create_dataset()
    """
    for x, y in training_dataset:
        print(x.numpy().shape)
        print(y.numpy().shape)
        break
    """
    train(dataset = training_dataset, configs = configs)

if __name__ == "__main__":
    main()
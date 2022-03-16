import tensorflow as tf
import tensorflow_addons as tfa

import src.dataset as dataset
import src.config as config

from src.utils import visualize
from src.models.nn import build_model
from src.losses.loss import BoxIouLoss, FocalLoss, HuberLoss
from src.utils.label_utils import _generate_anchors, _compute_gt

MIXED_PRECISION = False

if __name__=="__main__":
    
    if MIXED_PRECISION:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    configs = config.Configs(
        training_type="obd",
        dataset_path="datasets/data/VOC2012",
        training_dir="none")
    file_names = dataset.load_data(
        configs=configs,
        dataset_type="labeled")
    labeled_dataset = dataset.Dataset(
        file_names=file_names,
        configs=configs,
        dataset_type="labeled").create_dataset()

    # Training configurations
    EPOCHS = 10
    STEPS_PER_EPOCH = int(len(file_names) / configs.batch_size)
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=TOTAL_STEPS,
        decay_rate=0.96)
    optimizer = tfa.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=4e-5)
    if MIXED_PRECISION:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    """
    model = EfficientDet(
        num_classes=configs.num_classes,
        configs=configs,
        weights="imagenet")
    """
    anchors = _generate_anchors(
        configs, 
        configs.image_dims[0])

    model = build_model(
        0,
        configs,
        score_threshold = 0.25,
        num_classes = configs.num_classes,
        num_anchors = 9)

    focal_loss = FocalLoss()
    huber_loss = HuberLoss()
    iou_loss = BoxIouLoss(
        iou_loss_type="ciou",
        anchors=anchors)

    print(f"Num classes {configs.num_classes}")

    # Training Function
    @tf.function()
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            (label_cls, label_bbx) = labels
            logits_cls, logits_bbx = model(images, training=True)

            loss_cls = focal_loss(
                y_true=label_cls,
                y_pred=tf.cast(logits_cls, tf.float32))
            loss_bbx = huber_loss(
                y_true=label_bbx,
                y_pred=tf.cast(logits_bbx, tf.float32))
            loss_iou = iou_loss(
                y_true=label_bbx,
                y_pred=tf.cast(logits_bbx, tf.float32))
            loss = tf.reduce_sum([loss_cls, loss_bbx, loss_iou])
            if MIXED_PRECISION:
                loss = optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(
            target=loss, 
            sources=model.trainable_variables)
        if MIXED_PRECISION:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(
            grads_and_vars=zip(gradients, model.trainable_variables))
        return loss, loss_cls, loss_bbx, loss_iou

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")    
        for step, (images, label_cls, label_bbx) in enumerate(labeled_dataset):
            labels = (label_cls, label_bbx)

            labels = _compute_gt(
                images=images, 
                ground_truth=labels, 
                anchors=anchors, 
                num_classes=configs.num_classes)
            loss_vals = train_step(images, labels)
            
            print(
                f"Epoch {epoch} Step {step}/{STEPS_PER_EPOCH} ", \
                " ".join(f"loss-{i} {loss}" for i, loss in enumerate(loss_vals)))
            
        tf.keras.models.save_model(
            model,
            "model")


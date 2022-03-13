import albumentations as A
import tensorflow as tf
import tensorflow_addons as tfa

import src.dataset as dataset
import src.config as config

from src.test_models.efficientdet import EfficientDet
from src.losses.loss import FocalLoss, HuberLoss
from src.utils.label_utils import _generate_anchors, _compute_gt

if __name__=="__main__":
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
    
    focal_loss = FocalLoss()
    huber_loss = HuberLoss()

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
    
    
    model = EfficientDet(
        num_classes=configs.num_classes,
        D=0,
        weights="imagenet")

    print(f"Num classes {configs.num_classes}")

    # Training Function
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            (label_cls, label_bbx) = labels
            logits_cls, logits_bbx = model(images, training=True)

            loss_cls = focal_loss(
                y_true=label_cls,
                y_pred=logits_cls)
            loss_bbx = huber_loss(
                y_true=label_bbx,
                y_pred=logits_bbx)
            loss = tf.reduce_sum([loss_cls, loss_bbx])

        gradients = tape.gradient(
            target=loss, 
            sources=model.trainable_variables)
        optimizer.apply_gradients(
            grads_and_vars=zip(gradients, model.trainable_variables))
        return loss 

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")    
        for step, (images, label_cls, label_bbx) in enumerate(labeled_dataset):
            labels = (label_cls, label_bbx)
            anchors = _generate_anchors(
                configs, configs.image_dims[0])
            labels = _compute_gt(
                images=images, 
                classes=labels, 
                anchors=anchors, 
                num_classes=configs.num_classes)
            loss = train_step(images, labels)
            
            print(f"Epoch {epoch} Step {step}/{STEPS_PER_EPOCH} Loss {loss}")

        tf.keras.models.save_model(
            model,
            "model")

            

    

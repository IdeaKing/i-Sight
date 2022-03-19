import tensorflow as tf
import tensorflow_addons as tfa

import src.dataset as dataset
import src.config as config

from src.models.efficientdet import get_efficientdet
from src.losses.loss import EffDetLoss

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

    inputs = tf.keras.layers.Input((*configs.image_dims, 3))
    efficientdet_cls, efficientdet_bbx = get_efficientdet(
        name="efficientdet_d0",
        num_classes=configs.num_classes)(inputs)
    model = tf.keras.models.Model(
        inputs=[inputs], outputs=[efficientdet_cls, efficientdet_bbx])
    

    effdet_loss = EffDetLoss(
        num_classes=configs.num_classes,
        include_iou="ciou")


    print(f"Num classes {configs.num_classes}")

    # Training Function
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = effdet_loss(y_true=labels, y_pred=logits)
            if MIXED_PRECISION:
                loss = optimizer.get_scaled_loss(loss)

        gradients = tape.gradient(
            target=loss, 
            sources=model.trainable_variables)
        if MIXED_PRECISION:
            gradients = optimizer.get_unscaled_gradients(gradients)
        optimizer.apply_gradients(
            grads_and_vars=zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")    
        for step, (images, label_cls, label_bbx) in enumerate(labeled_dataset):
            labels = (label_cls, label_bbx)

            loss = train_step(images, labels)

            print(
                f"Epoch {epoch} Step {step}/{STEPS_PER_EPOCH} ", \
                f"loss {loss}") #" ".join(f"loss-{i} {loss}" for i, loss in enumerate(loss_vals)))
            
        tf.keras.models.save_model(
            model,
            "model")

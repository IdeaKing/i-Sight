import tensorflow as tf
import tensorflow_addons as tfa

from src.losses import loss 
from src.models import effdet
from src.models import temp_hparams
from src import dataset
from src import config
from src.utils import label_utils

if __name__ == "__main__":
    configs = temp_hparams.default_detection_configs()

    conf = config.Configs(
        training_type="obd", 
        dataset_path="datasets/data/VOC2012", 
        training_dir="test")

    model = effdet.EfficientDetNet(
        model_name="efficientdet-d0",
        config=configs)

    file_names = dataset.load_data(
        configs=conf,
        dataset_type="labeled")
    training_data = dataset.Dataset(
        file_names=file_names,
        configs=conf,
        dataset_type="labeled").create_dataset()

    
    focal_loss = loss.FocalLoss()
    huber_loss = loss.HuberLoss()

    # Training configurations
    EPOCHS = 10
    STEPS_PER_EPOCH = int(len(file_names) / conf.batch_size)
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=TOTAL_STEPS,
        decay_rate=0.96)
    optimizer = tfa.optimizers.AdamW(
        learning_rate=lr_schedule,
        weight_decay=4e-5)

    # Training Function
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            (label_cls, label_bbx) = labels
            logits_cls, logits_bbx = model(images, training=True)
            print("predicted cls", len(logits_cls))
            print("predicted bbx", len(logits_bbx))
            # print("predicted cls", logits_cls.numpy().shape)
            # print("predicted bbx", logits_bbx.numpy().shape)
            # print("gt cls", label_cls.numpy().shape)
            # print("gt bbx", label_bbx.numpy().shape)

            loss_cls = focal_loss(
                y_true=label_cls,
                y_pred=tf.cast(logits_cls[0], tf.float32))
            loss_bbx = huber_loss(
                y_true=label_bbx,
                y_pred=tf.cast(logits_bbx, tf.float32))
            loss = tf.reduce_sum([loss_cls, loss_bbx])


        gradients = tape.gradient(
            target=loss, 
            sources=model.trainable_variables)
        optimizer.apply_gradients(
            grads_and_vars=zip(gradients, model.trainable_variables))
        return loss 

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch}")    
        for step, (images, (label_cls, label_bbx)) in enumerate(training_data):
            labels = (label_cls, label_bbx)
            anchors = label_utils._generate_anchors(
                conf, 
                conf.image_dims[0])
            labels = label_utils._compute_gt(
                images=images, 
                ground_truth=labels, 
                anchors=anchors, 
                num_classes=configs.num_classes)
            loss = train_step(images, labels)
            
            print(f"Epoch {epoch} Step {step}/{STEPS_PER_EPOCH} Loss {loss}")

        tf.keras.models.save_model(
            model,
            "model")
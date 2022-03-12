import tensorflow as tf
import tensorflow_addons as tfa

import src.dataset as dataset
import src.config as config

from src.models.efficientdet import model_builder
from src.testing.anchors import AnchorGenerator, anchor_targets_bbox

def compute_gt(images, annots, anchors, num_classes):

    labels = annots[0]
    boxes = annots[1]

    target_reg, target_clf = anchor_targets_bbox(
            anchors, images, boxes, labels, num_classes)

    return images, (target_reg, target_clf)

def _generate_anchors(anchors_config,
                      im_shape: int):

    anchors_gen = [AnchorGenerator(
            size=anchors_config.sizes[i - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.downsampling_strides[i - 3]) 
            for i in range(3, 8)]

    shapes = [im_shape // (2 ** x) for x in range(3, 8)]

    anchors = [g((size, size, 3))
               for g, size in zip(anchors_gen, shapes)]

    return tf.concat(anchors, axis=0)


class EfficientDetFocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha: float = 0.25, gamma: float = 1.5) -> None:
        super(EfficientDetFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.loss_fn = tfa.losses.SigmoidFocalCrossEntropy(
            alpha=self.alpha, gamma=self.gamma,
            reduction=tf.losses.Reduction.SUM)
        
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        anchors_states = y_true[:, :, -1]
        labels = y_true[:, :, :-1]

        not_ignore_idx = tf.where(tf.not_equal(anchors_states, -1.))
        true_idx = tf.where(tf.equal(anchors_states, 1.))

        normalizer = tf.shape(true_idx)[0]
        normalizer = tf.cast(normalizer, tf.float32)
        normalizer = tf.maximum(tf.constant(1., dtype=tf.float32), normalizer)

        y_true = tf.gather_nd(labels, not_ignore_idx)
        y_pred = tf.gather_nd(y_pred, not_ignore_idx)


        return tf.divide(self.loss_fn(y_true, y_pred), normalizer)


class EfficientDetHuberLoss(tf.keras.losses.Loss):

    def __init__(self, delta: float = 1.) -> None:
        super(EfficientDetHuberLoss, self).__init__()
        self.delta = delta

        self.loss_fn = tf.losses.Huber(
            reduction=tf.losses.Reduction.SUM, 
            delta=self.delta)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        anchors_states = y_true[:, :, -1]
        labels = y_true[:, :, :-1]

        true_idx = tf.where(tf.equal(anchors_states, 1.))

        normalizer = tf.shape(true_idx)[0]
        normalizer = tf.cast(normalizer, tf.float32)
        normalizer = tf.maximum(tf.constant(1., dtype=tf.float32), normalizer)
        normalizer = tf.multiply(normalizer, tf.constant(4., dtype=tf.float32))

        y_true = tf.gather_nd(labels, true_idx)
        y_pred = tf.gather_nd(y_pred, true_idx)


        return 50. * tf.divide(self.loss_fn(y_true, y_pred), normalizer)

if __name__=="__main__":
    tf.config.run_functions_eagerly(True)


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
    
    focal_loss = EfficientDetFocalLoss()
    huber_loss = EfficientDetHuberLoss()

    # Training configurations
    EPOCHS = 10
    STEPS_PER_EPOCH = int(len(file_names) / configs.batch_size)
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=TOTAL_STEPS,
        decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    
    model = model_builder(configs, name="test")

    print(f"Num classes {configs.num_classes}")

    # Training Function
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            (label_bbs, label_cls) = labels
            logits_bbs, logits_cls = model(images, training=True)

            # print(f"BBS logits {logits_bbs.numpy()}")
            # print(f"CLS logits {logits_cls.numpy()}")
            # print(f"BBS labels {label_bbs.numpy()}")
            # print(f"CLS labels {label_cls.numpy()}")

            loss_cls = focal_loss(
                y_true=label_cls,
                y_pred=logits_cls)
            loss_bbs = huber_loss(
                y_true=label_bbs,
                y_pred=logits_bbs)
            # print(f"Loss cls {loss_cls}")
            # print(f"Loss bbs {loss_bbs}")
            loss = tf.reduce_sum([loss_cls, loss_bbs])

        gradients = tape.gradient(
            target=loss, 
            sources=model.trainable_variables)
        optimizer.apply_gradients(
            grads_and_vars=zip(gradients, model.trainable_variables))
        return loss 

    for epochs in range(EPOCHS):
        print(f"epoch {epochs}")    
        for step, (image, annots) in enumerate(labeled_dataset):
            anchors = _generate_anchors(configs, configs.image_dims[0])
            images, annots = compute_gt(
                images=image, 
                annots=annots, 
                anchors=anchors, 
                num_classes=configs.num_classes)
            loss = train_step(image, annots)
            
            print(f"Step {step} loss {loss}")
            # break
        tf.keras.models.save_model(
            model,
            "model")
        # break

            

    

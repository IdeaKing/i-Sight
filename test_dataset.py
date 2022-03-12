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
    configs = config.Configs(
        training_type="obd",
        dataset_path="datasets/data/obd_fundus",
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

    for image, (label, bbs) in labeled_dataset:
        print(f"Image shape: {image.numpy().shape}")
        print(f"Label shape: {label.numpy().shape}")
        print(f"BBS shape: {bbs.numpy().shape}")

        print(f"Labels {label})")
        print(f"BBS {bbs}")

        annots = (label, bbs)

        anchors = _generate_anchors(configs, configs.image_dims[0])
        images, (target_reg, target_clf) = compute_gt(
            images=image, 
            annots=annots, 
            anchors=anchors, 
            num_classes=configs.num_classes)
        
        print(f"Target bbs {target_reg.numpy()[0]}")
        print(f"Target cls {target_clf.numpy()[0]}")
        """
        print("Target regression boxes: {}".format(target_reg.numpy().shape))
        print("Target classifications: {}".format(target_clf.numpy().shape))
        
        rand_logits_boxes = tf.random.uniform(
            (4, 12276, 4), maxval=512
        )
        rand_logits_cls = tf.random.uniform(
            (4, 12276, 8), maxval=512
        )
        loss_box = focal_loss(y_true=target_reg, y_pred=rand_logits_boxes)
        print(loss_box)
        loss_cls = huber_loss(y_true=target_clf, y_pred=rand_logits_cls)
        print(loss_cls)
        """

        break

    

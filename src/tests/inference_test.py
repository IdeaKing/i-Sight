"""Inference script."""
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from src.utils.anchors import Anchors
from src.models.efficientdet import EfficientDet, get_efficientdet
from src.utils.label_utils import to_corners


def make_prediction(image,
                    max_output_size_per_class=100,
                    max_total_size=100,
                    iot_threshold=0.5,
                    score_threshold=0.1):
    box_variance = tf.cast(
        [0.1, 0.1, 0.2, 0.2], tf.float32
    )
    
    #padded_image, new_shape, scale = resize_and_pad(image, scale_jitter=None)
    anchor_boxes = Anchors(aspect_ratios=[0.5, 1, 2], scales=[0, 1/3, 2/3]).get_anchors(
        512, 512)

    preds = model.predict(tf.expand_dims(image, axis=0))

    boxes = preds[..., :4] * box_variance
    boxes = tf.concat(
        [
            boxes[..., :2] * anchor_boxes[..., 2:] + anchor_boxes[..., :2],
            tf.exp(boxes[..., 2:]) * anchor_boxes[..., 2:]
        ],
        axis=-1
    )
    boxes = to_corners(boxes)
    classes = tf.nn.sigmoid(preds[..., 4:])

    nms = tf.image.combined_non_max_suppression(
        tf.expand_dims(boxes, axis=2),
        classes,
        max_output_size_per_class=max_output_size_per_class,
        max_total_size=max_total_size,
        iou_threshold=iot_threshold,
        score_threshold=score_threshold,
        clip_boxes=False
    )

    valid_dets = nms.valid_detections[0]

    image = Image.fromarray(np.array(image * 255, dtype=np.uint8))
    draw = ImageDraw.Draw(image)

    for i in range(valid_dets):
        print(nms.nmsed_classes[0, i])
        print(nms.nmsed_scores[0, i])
        x_min, y_min, x_max, y_max = nms.nmsed_boxes[0, i] #/ scale
        boxes = (x_min, y_min, x_max, y_max)
        draw.rectangle(boxes, outline=(255, 0, 0), width=2)

    plt.imshow(image)
    plt.show()



model = EfficientDet()
model.build((None, 512, 512, 3))
model.load_weights("models/efficientdet-14.h5")

raw_image = tf.io.read_file("datasets/data/VOC2012/images/2007_002266.jpg")
image = tf.image.decode_image(raw_image, channels=3)
image = tf.image.resize(image, (512, 512))
image = tf.cast(image, tf.float32) / 255.0

make_prediction(image)
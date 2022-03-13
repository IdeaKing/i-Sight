import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils.postprocess import FilterDetections
from src.utils.visualize import draw_boxes

import src.utils.training_utils as t_utils


class Configs:
    dataset_path = "datasets/data/VOC2012"
    image_dims = (512, 512)
    labels_path = os.path.join(
        dataset_path, 
        "labels.txt")
    labels = t_utils.parse_label_file(labels_path)
    num_classes = len(labels)

    score_threshold = 0.35
    iou_threshold = 0.01
    max_box_num = 100
    anchors = 9
    ratios = [0.5, 1, 2]
    scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
    downsampling_strides = [8, 16, 32, 64, 128]
    sizes = [32, 64, 128, 256, 512]


def preprocess_image(image_path, configs):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255 # Normalize
    image = cv2.resize(image, configs.image_dims)
    image = np.array(image, np.float32)
    return image


def test(image_path, model, configs):
    image = np.expand_dims(
        preprocess_image(image_path, configs),
        axis=0) # (1, 512, 512, 3)
    pred_box, pred_cls = model(image, training=False)
    boxes, labels, scores = FilterDetections(
        configs, Configs.score_threshold)(images=image,
            regressors=pred_box,
            class_scores=pred_cls)
    labels = [list(configs.labels.keys())[int(l)] 
              for l in labels[0]]
    
    scores = scores[0]
    boxes = boxes[0]

    image = draw_boxes(
        np.squeeze(image, axis=0),
        boxes,
        labels, 
        scores)

    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    configs = Configs()

    # Path to image and model
    image_path = "datasets/data/VOC2012/images/2009_002744.jpg"
    model_path = "model"
    model = tf.keras.models.load_model(model_path)

    # Test the model on the image
    test(image_path=image_path,
         model=model,
         configs=configs)

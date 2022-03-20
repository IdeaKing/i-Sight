import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils.postprocess import FilterDetections
from src.utils.visualize import draw_boxes

import src.utils.training_utils as t_utils


def preprocess_image(image_path, image_dims):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image/255  # Normalize
    image = cv2.resize(image, image_dims)
    image = np.array(image, np.float32)
    return image


def test(image_path, model, image_dims, label_dict, score_threshold):
    image = np.expand_dims(
        preprocess_image(image_path, image_dims),
        axis=0)  # (1, 512, 512, 3)

    pred_cls, pred_box = model(image, training=False)
    labels, bboxes, scores = FilterDetections(score_threshold)(
        labels=pred_cls,
        bboxes=pred_box)

    labels = [list(label_dict.keys())[int(l)]
              for l in labels[0]]
    bboxes = bboxes[0]
    scores = scores[0]

    image = draw_boxes(
        image=np.squeeze(image, axis=0),
        bboxes=bboxes,
        labels=labels,
        scores=scores)

    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    image_dims = (512, 512)
    label_dict = t_utils.parse_label_file(
        path_to_label_file="datasets/data/VOC2012/labels.txt")
    score_threshold = 0.10

    # Path to image and model
    image_path = "datasets/data/VOC2012/images/2007_002545.jpg"
    model_path = "model/model"
    model = tf.keras.models.load_model(model_path)

    # Test the model on the image
    test(image_path=image_path,
         model=model,
         image_dims=image_dims,
         label_dict=label_dict,
         score_threshold=score_threshold)

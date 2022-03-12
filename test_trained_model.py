import os
import cv2
import numpy as np
import tensorflow as tf

import src.utils.training_utils as t_utils

from src.models.efficientdet import BoxTransform, ClipBoxes, MapToInputImage
from src.losses.anchor import Anchors
from src.utils.nms import NMS


class Configs:
    dataset_path = "datasets/data/obd_fundus"
    image_dims = (256, 256)
    labels_path = os.path.join(
        dataset_path, "labels.txt")
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

def fundus_preprocessing(image):
    """An improved luminosity and contrast enhancement 
    framework for feature preservation in color fundus images"""
    # RGB to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    H,S,V = cv2.split(image_hsv)
    # Gamma adjustment on V Channe;
    gamma_val = 1/2.2
    gamma_tab = [((i / 255) ** gamma_val) * 255 for i in range(256)]
    table = np.array(gamma_tab, np.uint8)
    gamma_V = cv2.LUT(V, table)
    image_hsv_gamma = cv2.merge((H, S, gamma_V))
    # HSV to RGB to LAB
    image_lab = cv2.cvtColor(
        cv2.cvtColor(
            image_hsv_gamma, cv2.COLOR_HSV2RGB),
        cv2.COLOR_RGB2LAB)
    # CLAHE on L Channel
    l,a,b = cv2.split(image_lab)
    clahe = cv2.createCLAHE(
        clipLimit=2.5, 
        tileGridSize=(8,8))
    l = clahe.apply(l)
    lab_image = cv2.merge((l,a,b))
    # Convert image back to rgb
    image_processed = cv2.cvtColor(
        lab_image, cv2.COLOR_LAB2RGB)
    return image_processed

def preprocess_image(image_path, configs):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = fundus_preprocessing(image)
    image = image/255 # Normalize
    image = cv2.resize(image, configs.image_dims)
    image = np.array(image, np.float32)
    return image

def postprocess(outputs, configs):
    box_transform = BoxTransform()
    clip_boxes = ClipBoxes(configs)
    anchors_func = Anchors(configs)
    map_to_original = MapToInputImage(configs.image_dims, configs)
    nms = NMS(configs)

    anchors = anchors_func(image_size=configs.image_dims)
    reg_results, cls_results = outputs[..., :4], outputs[..., 4:]
    reg_results = np.expand_dims(reg_results, axis=0)
    cls_results = np.expand_dims(cls_results, axis=0)
    transformed_anchors = box_transform(anchors, reg_results)
    transformed_anchors = clip_boxes(transformed_anchors)
    transformed_anchors = map_to_original(transformed_anchors)
    scores = tf.math.reduce_max(cls_results, axis=2).numpy()
    classes = tf.math.argmax(cls_results, axis=2).numpy()
    final_boxes, final_scores, final_classes = nms(boxes=transformed_anchors[0, :, :],
                                                    box_scores=np.squeeze(scores),
                                                    box_classes=np.squeeze(classes))
    return final_boxes.numpy(), final_scores.numpy(), final_classes.numpy()

def draw_boxes_on_image(image, boxes, scores, classes, configs):
    num_boxes = boxes.shape[0]
    print("boxes", boxes)

    for i in range(num_boxes):
        cv2.rectangle(img=image, 
                      pt1=(boxes[i, 0], 
                      boxes[i, 1]), 
                      pt2=(boxes[i, 2], 
                      boxes[i, 3]), 
                      color=(255, 0, 0), 
                      thickness=2)
        """
        cv2.putText(img=image, 
                    text=class_and_score, 
                    org=(boxes[i, 0], boxes[i, 1] - 10), 
                    fontFace=cv2.FONT_HERSHEY_COMPLEX, 
                    fontScale=1.5, 
                    color=(0, 255, 255), 
                    thickness=2)
        """
    cv2.imshow("Detections", image)
    cv2.waitKey(0)

def test(image_path, model, configs):
    print("Preprocessing image")
    image = preprocess_image(image_path, configs)
    image_for_model = np.expand_dims(image, axis=0)
    print("Running inference")
    outputs = model(image_for_model, training=False)
    outputs = np.squeeze(outputs, axis=0)
    print("Post processing")
    boxes, scores, classes = postprocess(outputs, configs)
    draw_boxes_on_image(image, boxes, scores, classes, configs)

if __name__ == "__main__":
    configs = Configs()

    # Path to image and model
    image_path = "datasets\data\obd_fundus\images\cws_0006.jpg"# "datasets/data/test_fundus/test.jpg"
    model_path = "test-new-training/ema"
    model = tf.keras.models.load_model(model_path)

    test(image_path=image_path,
         model=model,
         configs=configs)

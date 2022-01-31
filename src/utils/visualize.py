# i-Sight Display Functions for Object Detection

import matplotlib.pyplot as plt
import cv2


def show_boxes(image, boxes):
    """Boxes should be [num boxes, x min, y min, x max, y max, label] """
    for box in boxes:
        # [xmin, ymin, xmax, ymax, l]
        xmin, ymin, xmax, ymax, l = box
        xmin = int(image.shape[1] * xmin)
        xmax = int(image.shape[1] * xmax)
        ymin = int(image.shape[0] * ymin)
        ymax = int(image.shape[0] * ymax)
        image = cv2.rectangle(
            image, 
            (xmin, ymin),
            (xmax, ymax),
            (255, 255, 255),
            3)
    plt.imshow(image)
    plt.show()
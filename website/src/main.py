import os
import shutil
import random
import datetime

import cv2

import numpy as np
import tensorflow as tf

from . import fundus
from .utils import file_reader, image_output

def predict(fundus_image: np.array, 
            oct_image: np.array, 
            models: dict,
            save_dir: str = "user_saves",
            path_to_labels: str = "models/labels.txt") -> dict:
    """Runs the prediction on the input images."""
    
    # Save the input images to user folder
    date_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    salt = str(random.randint(0, 100000)).zfill(6)
    dir_name = str(date_time + "_" + salt)
    save_loc = os.path.join(save_dir, dir_name)

    # Read the labels
    label_dict = file_reader.parse_label_file(path_to_labels)
    
    # Run the Fundus prediction
    fundus_predictions = fundus.Fundus(image=fundus_image, label_dict=label_dict)(
        detection_model=models["object_detection"], 
        segmentation_model=models["retina_segmentation"])
    
    # Temporary OCT data
    if oct_image is not None:
        oct_prediction = {"main_image": cv2.cvtColor(cv2.imread("docs/oct.jpg"), cv2.COLOR_BGR2RGB),
                          "found": True}
    else:
        oct_prediction = {"main_image": cv2.cvtColor(cv2.imread("docs/oct_not_provided.jpg"), cv2.COLOR_BGR2RGB),
                          "found": False}
        oct_image = oct_prediction["main_image"]

    output = image_output.create_output_image(
        fundus_predictions=fundus_predictions,
        oct_predictions=oct_prediction)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    if not os.path.exists(save_loc):
        os.mkdir(save_loc)
    else:
        raise Exception("Save directory already exists")
    cv2.imwrite(
        os.path.join(save_loc, "input_fundus.jpg"),
        cv2.cvtColor(fundus_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        os.path.join(save_loc, "input_oct.jpg"),
        cv2.cvtColor(oct_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(
        os.path.join(save_loc, "output.jpg"),
        output)
    return output
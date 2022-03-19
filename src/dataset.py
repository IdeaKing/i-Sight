# Thomas Chia i-Sight Dataset Pipeline

import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

from src.utils.anchors import Encoder
from src.utils import label_utils 

# The full tf.data pipline
class Dataset():
    def __init__(self, file_names, configs, dataset_type):
        self.file_names = file_names
        self.augment_ds = False
        self.configs = configs
        self.image_dims = configs.image_dims
        self.dataset_type = dataset_type
        self.encoder = Encoder()

    def parse_augment_image(self, file_name):
        """For augmenting images and bboxes."""
        # Read and preprocess the image
        file_name = bytes.decode(file_name, encoding="utf-8")
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_dims)
        image = self.fundus_preprocessing(image)
        # Augmentation function
        transform = A.Compose(
            [A.Flip(p = 0.5),
             A.Rotate(p = 0.5),
             A.ElasticTransform(p = 0.25),
             A.OpticalDistortion(p = 0.25)])
        aug = transform(
            image = image)
        image = aug["image"]/255.
        image = np.array(image, dtype=np.float32)
        return image


    def augment(self, image, label, bbx, augment=True):
        """For augmenting images and bboxes."""
        # Read and preprocess the image
        image, label, bbx = (image.numpy(), label.numpy().tolist(), bbx.numpy().tolist())
        # Augmentation function
        if augment:
            transform = A.Compose(
                [A.Flip(p=0.5),
                A.Rotate(p=0.5),
                A.RandomBrightnessContrast(p=0.2)],
                bbox_params=A.BboxParams(
                    format="pascal_voc", 
                    label_fields=["class_labels"]))
        else:
            transform = A.Compose(
                [],
                bbox_params=A.BboxParams(
                    format="pascal_voc", 
                    label_fields=["class_labels"]))
        aug = transform(
            image=image,
            bboxes=bbx,
            class_labels=label)
        image = np.array(aug["image"], np.float32)
        labels = np.array(aug["class_labels"], np.int32)
        bbx = np.array(aug["bboxes"], np.float32)
        return image, labels, bbx


    def fundus_preprocessing(self, image):
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


    def parse_xml(self, path_to_label):
        """Parses the PascalVOC XML Type file."""
        # Reads a voc annotation and returns
        # a list of tuples containing the ground 
        # truth boxes and its respective label

        root = ET.parse(path_to_label).getroot()
        image_size = (int(root.findtext("size/width")),
                    int(root.findtext("size/height")))  
        boxes = root.findall("object")
        bbx = []
        labels = []

        for b in boxes:
            bb = b.find("bndbox")
            bb = (int(bb.findtext("xmin")),
                int(bb.findtext("ymin")),
                int(bb.findtext("xmax")),
                int(bb.findtext("ymax")))
            bbx.append(bb)
            labels.append(
                int(self.configs.labels[b.findtext("name")]))

        bbx = tf.stack(bbx)
        # bbx are in relative mode
        bbx = label_utils.to_relative(bbx, image_size) 
        # Scale bbx to input image dims
        bbx = label_utils.to_scale(bbx, self.configs.image_dims)
        return labels, bbx


    def parse_process_image(self, file_name):
        file_name = bytes.decode(file_name, encoding="utf-8")
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.fundus_preprocessing(image)
        image = image/255 # Normalize
        image = cv2.resize(image, self.image_dims)
        image = np.array(image, np.float32)
        return image
    

    def parse_process_voc(self, file_name):
        file_name = bytes.decode(file_name, encoding="utf-8")
        labels, bbx = self.parse_xml(file_name)
        labels = np.array(labels, np.int32)
        bbx = np.array(bbx, np.float32)
        return labels, bbx


    def create_dataset(self):
        image_list_ds = tf.data.Dataset.from_tensor_slices([
            os.path.join(
                self.configs.dataset_path,
                self.configs.images_dir,
                file + ".jpg") for file in self.file_names])
        label_list_ds = tf.data.Dataset.from_tensor_slices([
            os.path.join(
                self.configs.dataset_path,
                self.configs.labels_dir,
                file + ".xml") for file in self.file_names])
        l_image_ds = image_list_ds.map(
            lambda x: tf.numpy_function(
                self.parse_process_image,
                inp=[x],
                Tout=tf.float32,
                name="l_parse_image"),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            name = "l_image_map_ds")
        l_label_ds = label_list_ds.map(
            lambda x: tf.numpy_function(
                self.parse_process_voc,
                inp=[x],
                Tout=[tf.int32, tf.float32]),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
            name="bbox_map_ds")
        ds = tf.data.Dataset.zip((l_image_ds, l_label_ds))
        if self.augment_ds:
            ds = ds.map(
                lambda x, y: tf.py_function(
                    self.augment,
                    inp=[x, *y],
                    Tout=[tf.float32, tf.int32, tf.float32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                name="augment_ds")
        else:
            ds = ds.map(
                lambda x, y: tf.py_function(
                    self.augment,
                    inp=[x, *y, False],
                    Tout=[tf.float32, tf.int32, tf.float32]),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                name="none_augment_ds")
        ds = ds.shuffle(self.configs.shuffle_size)
        ds = ds.padded_batch(batch_size=self.configs.batch_size,
                             padded_shapes=((*self.configs.image_dims, 3), (None,), (None, 4)),
                             padding_values=(0., -1, -1.))
        ds = ds.map(
            map_func=self.encoder.encode_batch, 
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return ds

def load_data(configs, dataset_type = "labeled"):
    """Find the file paths for training and validation off of file."""
    file_names = []
    if dataset_type == "labeled":
        with open(
            os.path.join(
                configs.dataset_path, "labeled_train.txt")) as reader:
            for line in reader.readlines():
                file_names.append(line.rstrip().split(" ")[0])
    elif dataset_type == "unlabeled":
        with open(
            os.path.join(
                configs.dataset_path, "unlabeled_train.txt")) as reader:
            for line in reader.readlines():
                file_names.append(line.rstrip().split(" ")[0])
    elif dataset_type == "student":
        with open(
            os.path.join(
                configs.dataset_path, "student_train.txt")) as reader:
            for line in reader.readlines():
                file_names.append(line.rstrip().split(" ")[0])
    random.shuffle(file_names)
    return file_names
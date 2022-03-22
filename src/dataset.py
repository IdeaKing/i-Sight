# Thomas Chia i-Sight Dataset Pipeline

import os
import random
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

from typing import List, Tuple

from src.utils.anchors import Encoder
from src.utils import label_utils


class Dataset():
    def __init__(self,
                 file_names: List,
                 dataset_path: str,
                 labels_dict: dict,
                 training_type: str,
                 scales: Tuple,
                 aspect_ratios: Tuple,
                 batch_size: int = 4,
                 shuffle_size: int = 64,
                 images_dir: str = "images",
                 labels_dir: str = "labels",
                 image_dims: Tuple = (512, 512),
                 augment_ds: bool = False,
                 dataset_type: str = "labeled"):
        """ Creates the object detection dataset. """
        self.file_names = file_names
        self.dataset_path = dataset_path
        self.labels_dict = labels_dict
        self.training_type = training_type
        self.batch_size = batch_size
        self.shuffle_size = shuffle_size
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_dims = image_dims
        self.augment_ds = augment_ds
        self.dataset_type = dataset_type
        self.encoder = Encoder(scales=scales, 
                               aspect_ratios=aspect_ratios)

    def randaug(self, image):
        """For augmenting images and bboxes. Based on AutoAugment"""
        # Read and preprocess the image
        # Augmentation function
        transform = A.Compose(
            [A.Flip(p=0.5),
             A.Rotate(p=0.5),
             A.ElasticTransform(p=0.5),
             A.OpticalDistortion(p=0.5),
             A.CoarseDropout(p=0.5),
             A.GaussianBlur(p=0.25),
             A.RandomSunFlare(p=0.20),
             A.ImageCompression(p=0.25),
             A.RandomBrightnessContrast(p=0.2)])
        aug = transform(image=image)
        return np.array(aug["image"], np.float32)

    def augment(self, image, label, bbx):
        """For augmenting images and bboxes."""
        # Read and preprocess the image
        image, label, bbx = (image, label.tolist(), bbx.tolist())
        # Augmentation function
        if self.augment_ds:
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
        H, S, V = cv2.split(image_hsv)
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
        l, a, b = cv2.split(image_lab)
        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab_image = cv2.merge((l, a, b))
        # Convert image back to rgb
        image_processed = cv2.cvtColor(
            lab_image, cv2.COLOR_LAB2RGB)
        return image_processed

    def parse_process_voc(self, file_name):
        """Parses the PascalVOC XML Type file."""
        # Reads a voc annotation and returns
        # a list of tuples containing the ground
        # truth boxes and its respective label
        source = open(file_name)
        root = ET.parse(source).getroot()
        image_size = (int(root.findtext("size/width")),
                      int(root.findtext("size/height")))
        boxes = root.findall("object")
        bbx = []
        labels = []

        for b in boxes:
            bb = b.find("bndbox")
            bb = (int(float(bb.findtext("xmin"))),
                  int(float(bb.findtext("ymin"))),
                  int(float(bb.findtext("xmax"))),
                  int(float(bb.findtext("ymax"))))
            bbx.append(bb)
            labels.append(
                int(self.labels_dict[b.findtext("name")]))
        bbx = tf.stack(bbx)
        # bbx are in relative mode
        bbx = label_utils.to_relative(bbx, image_size)
        # Scale bbx to input image dims
        bbx = label_utils.to_scale(bbx, self.image_dims)
        source.close()
        return np.array(labels), np.array(bbx)

    def parse_process_image(self, file_name):
        image = tf.io.read_file(file_name)
        image = tf.io.decode_jpeg(
            image,
            channels=3)
        # image = self.fundus_preprocessing(image)
        image = tf.cast(image, tf.float32)/255. # Normalize
        image = tf.image.resize(images=image,
                                size=self.image_dims)
        image = np.asarray(image, np.float32)
        return image

    def parse_object_detection(self, file_name):
        file_name = bytes.decode(file_name, encoding="utf-8")
        image_file_path = os.path.join(self.dataset_path,
                                       self.images_dir,
                                       file_name + ".jpg")
        label_file_path = os.path.join(self.dataset_path,
                                       self.labels_dir,
                                       file_name + ".xml")
        image = self.parse_process_image(
            file_name=image_file_path)
        label, bboxs = self.parse_process_voc(
            file_name=label_file_path)
        image, label, bboxs = self.augment(
            image=image, label=label, bbx=bboxs)
        bboxs = label_utils.to_xywh(bboxs)
        image, label, bboxs = (np.array(image, np.float32),
                               np.array(label, np.int32),
                               np.array(bboxs, np.float32))
        label, bboxs = self.encoder._encode_sample(
            image_shape=self.image_dims,
            gt_boxes=bboxs,
            classes=label)
        return image, label, bboxs

    def parse_unlabeled_images(self, file_name):
        file_name = bytes.decode(file_name, encoding="utf-8")
        image_file_path = os.path.join(self.dataset_path,
                                       self.images_dir,
                                       file_name + ".jpg")
        image = self.parse_process_image(file_name=image_file_path)
        # image = self.fundus_preprocessing(image)
        aug_image = self.randaug(image=image)
        return image, aug_image

    def __call__(self):
        list_ds = tf.data.Dataset.from_tensor_slices(self.file_names)
        if self.dataset_type == "labeled":
            if self.training_type == "object_detection":
                ds = list_ds.map(
                    lambda x: tf.numpy_function(
                        self.parse_object_detection,
                        inp=[x],
                        Tout=[tf.float32, tf.float32, tf.float32]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    name="object_detection_parser")
                ds = ds.shuffle(self.shuffle_size)
                ds = ds.batch(self.batch_size)
                ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
                # ds = ds.cache()
                return ds
        elif self.dataset_type == "unlabeled":
            ds = list_ds.map(
                    lambda x: tf.numpy_function(
                        self.parse_unlabeled_images,
                        inp=[x],
                        Tout=[tf.float32, tf.float32]),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE,
                    name="unlabeled_parser")
            ds = ds.shuffle(self.shuffle_size)
            ds = ds.batch(self.batch_size)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
            # ds = ds.cache()
            return ds
        else:
            ValueError(f"{self.dataset_type} isn't a valid dataset type.")


def load_data(dataset_path, file_name="labeled_train.txt"):
    """Reads each line of the file."""
    file_names = []
    with open(
        os.path.join(
            dataset_path, file_name)) as reader:
        for line in reader.readlines():
            file_names.append(line.rstrip().split(" ")[0])
    random.shuffle(file_names)
    return file_names

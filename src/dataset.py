# Thomas Chia i-Sight Dataset Pipeline

import os
import random
import xml.dom.minidom as xdom

import cv2
import numpy as np
import tensorflow as tf
import albumentations as A

# The full tf.data pipline
class Dataset():
    def __init__(
        self, 
        file_names, 
        configs,
        dataset_type):
        self.file_names = file_names
        self.configs = configs
        self.image_dims = configs.image_dims
        self.dataset_type = dataset_type


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
        image = np.array(aug["image"]/255)
        return image


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
        obj_and_box_list = []
        contents = xdom.parse(path_to_label)
        annotation = contents.documentElement
        size = annotation.getElementsByTagName("size")
        image_height = 0
        image_width = 0
        # Find the width and height of the original image
        for s in size:
            image_height = int(
                s.getElementsByTagName(
                    "height")[0].childNodes[0].data)
            image_width = int(
                s.getElementsByTagName(
                    "width")[0].childNodes[0].data)
        obj = annotation.getElementsByTagName("object")
        # Find all of the bounding boxes
        for o in obj:
            o_list = []
            obj_name = o.getElementsByTagName(
                "name")[0].childNodes[0].data
            bndbox = o.getElementsByTagName("bndbox")
            for box in bndbox:
                xmin = box.getElementsByTagName(
                    "xmin")[0].childNodes[0].data
                ymin = box.getElementsByTagName(
                    "ymin")[0].childNodes[0].data
                xmax = box.getElementsByTagName(
                    "xmax")[0].childNodes[0].data
                ymax = box.getElementsByTagName(
                    "ymax")[0].childNodes[0].data
                x_min = int(int(xmin)*image_width/self.configs.image_dims[0])
                y_min = int(int(ymin)*image_height/self.configs.image_dims[1])
                x_max = int(int(xmax)*image_width/self.configs.image_dims[0])
                y_max = int(int(ymax)*image_height/self.configs.image_dims[1])
                o_list.append(x_min)
                o_list.append(y_min)
                o_list.append(x_max)
                o_list.append(y_max)
                break
            o_list.append(int(self.configs.labels[obj_name]))
            obj_and_box_list.append(o_list)
        return obj_and_box_list


    def pad_bboxes(self, box):
        """Pads the bounding boxes for easy batching."""
        padding_boxes = self.configs.max_box_num \
            - len(box)
        # Append "zero boxes" to pad the list
        if padding_boxes > 0:
            for boxes in range(padding_boxes):
                box.append([0,0,0,0, -1])
        return box


    def parse_process_image(self, file_name):
        file_name = bytes.decode(file_name, encoding="utf-8")
        image = cv2.imread(file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_dims)
        image = self.fundus_preprocessing(image)
        image = image/255 # Normalize
        image = np.array(image, np.float32)
        return image
    

    def parse_process_voc(self, file_name):
        file_name = bytes.decode(file_name, encoding="utf-8")
        labels = self.parse_xml(file_name)
        labels = self.pad_bboxes(labels)
        labels = np.array(labels, np.float32)
        return labels
    

    def load_serialized_dataset(self):
        """Loads dataset from directory."""
        if self.dataset_type == "labeled":
            # Loads the dataset from directory
            l_image_ds = tf.data.experimental.load(
                self.configs.serialized_dir + "/labeled_im")
            l_label_ds = tf.data.experimental.save(
                self.configs.serialized_dir + "/labeled_lb")
            # Process and zip the datasets
            ds = tf.data.Dataset.zip((l_image_ds, l_label_ds))
            ds = ds.shuffle(
                buffer_size = self.configs.shuffle_size)
            ds = ds.batch(
                self.configs.batch_size)
            ds = ds.prefetch(
                buffer_size = tf.data.experimental.AUTOTUNE)
            return ds
        elif self.dataset_type == "unlabeled":
            # Loads the dataset from directory
            u_orgim_ds = tf.data.experimental.load(
                self.configs.serialized_dir + "/unlabeled_orgim")
            u_augim_ds = tf.data.experimental.save(
                self.configs.serialized_dir + "/unlabeled_augim")
            # Process and zip the datasets
            ds = tf.data.Dataset.zip((u_orgim_ds, u_augim_ds))
            ds = ds.shuffle(
                buffer_size = self.configs.shuffle_size)
            ds = ds.batch(
                self.configs.batch_size)
            ds = ds.prefetch(
                buffer_size = tf.data.experimental.AUTOTUNE)
            return ds


    def serialize_dataset(self, inp1, inp2):
        """Processes and saves the dataset."""
        if self.dataset_type == "labeled":
            # inp1 is the OrgImage Dataset
            tf.data.experimental.save(
                inp1, 
                self.configs.serialized_dir + "/labeled_im")
            # inp2 is the Labels Dataset
            tf.data.experimental.save(
                inp2,
                self.configs.serialized_dir + "/labeled_lb")
        elif self.dataset_type == "unlabeled":
            # inp1 is the OrgImage Dataset
            tf.data.experimental.save(
                inp1, 
                self.configs.serialized_dir + "/unlabeled_orgim")
            # inp2 is the AugImage Dataset
            tf.data.experimental.save(
                inp2,
                self.configs.serialized_dir + "/unlabeled_augim")


    def create_dataset(self):
        if self.dataset_type == "labeled" or self.dataset_type == "student":
            if self.dataset_type == "labeled":
                # List of the filepaths for images and labels
                images_paths = [
                    os.path.join(
                        self.configs.dataset_path,
                        self.configs.images_dir,
                        file + ".jpg") for file in self.file_names]
                labels_paths = [
                    os.path.join(
                        self.configs.dataset_path,
                        self.configs.labels_dir,
                        file + ".xml") for file in self.file_names]
            elif self.dataset_type == "student":
                images_paths = [
                    os.path.join(
                        self.configs.dataset_path,
                        self.configs.student_images_dir,
                        file + ".jpg") for file in self.file_names]
                labels_paths = [
                    os.path.join(
                        self.configs.dataset_path,
                        self.configs.student_labels_dir,
                        file + ".xml") for file in self.file_names]
            # Create the dataset object for the images
            image_list_ds = tf.data.Dataset.from_tensor_slices(images_paths)
            l_image_ds = image_list_ds.map(
                lambda x: tf.numpy_function(
                    self.parse_process_image,
                    inp=[x],
                    Tout=tf.float32,
                    name="l_parse_image"),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                name = "l_image_map_ds")
            # Create the dataset object for the labels
            label_list_ds = tf.data.Dataset.from_tensor_slices(labels_paths)
            l_label_ds = label_list_ds.map(
                lambda x: tf.numpy_function(
                    self.parse_process_voc,
                    inp=[x],
                    Tout=tf.float32,
                    name="l_parse_label"),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                name = "bbox_map_ds")
            # Zip dataset and apply training configs
            ds = tf.data.Dataset.zip((l_image_ds, l_label_ds))
            ds = ds.shuffle(
                buffer_size = self.configs.shuffle_size)
            if self.dataset_type == "student":
                ds = ds.batch(
                    self.configs.student_batch_size)
            else:
                ds = ds.batch(
                    self.configs.batch_size)
            ds = ds.prefetch(
                buffer_size = tf.data.experimental.AUTOTUNE)
            return ds
        elif self.dataset_type == "unlabeled":
            # List of the filepaths for images and labels
            images_paths = [
                os.path.join(
                    self.configs.dataset_path,
                    self.configs.images_dir,
                    file + ".jpg") for file in self.file_names]
            image_list_ds = tf.data.Dataset.from_tensor_slices(images_paths)
            u_orgim_ds = image_list_ds.map(
                lambda x: tf.numpy_function(
                    self.parse_process_image,
                    inp=[x],
                    Tout=tf.float32,
                    name="u_parse_orgim"),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                name = "u_org_map_ds")
            u_augim_ds = image_list_ds.map(
                lambda x: tf.numpy_function(
                    self.parse_augment_image,
                    inp=[x],
                    Tout=tf.float32,
                    name="u_parse_augim"),
                num_parallel_calls=tf.data.experimental.AUTOTUNE,
                name = "u_aug_map_ds")
            # Zip dataset and apply training configs
            ds = tf.data.Dataset.zip((u_orgim_ds, u_augim_ds))
            ds = ds.shuffle(
                buffer_size = self.configs.shuffle_size)
            ds = ds.batch(
                self.configs.unlabled_batch_size)
            ds = ds.prefetch(
                buffer_size = tf.data.experimental.AUTOTUNE)
            return ds

def get_files(configs):
    """Creates a list of the files."""
    ## Legacy func to be deleted.
    image_file = open(
        configs.dataset_path + "\labeled_train.txt", "r")
    image_file_names = image_file.readlines()
    image_file_names = [name.strip() for name in image_file_names]
    
    return image_file_names


def load_data(configs, dataset_type = "labeled"):
    """Find the file paths for training and validation off of file."""
    file_names = []
    if dataset_type == "labeled":
        with open(
            os.path.join(
                configs.dataset_path, 'labeled_train.txt')) as reader:
            for line in reader.readlines():
                file_names.append(line.rstrip().split(' ')[0])
    elif dataset_type == "unlabeled":
        with open(
            os.path.join(
                configs.dataset_path, 'unlabeled_train.txt')) as reader:
            for line in reader.readlines():
                file_names.append(line.rstrip().split(' ')[0])
    elif dataset_type == "student":
        with open(
            os.path.join(
                configs.dataset_path, 'student_train.txt')) as reader:
            for line in reader.readlines():
                file_names.append(line.rstrip().split(' ')[0])
    random.shuffle(file_names)
    return file_names
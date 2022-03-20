import numpy as np
import tensorflow as tf

import src.dataset as dataset
import src.config as config
import src.utils.postprocess as post

if __name__=="__main__":
    configs = config.Configs(
        training_type="obd",
        dataset_path="datasets/data/VOC2012",
        training_dir="none")
    file_names = dataset.load_data(
        dataset_path=configs.dataset_path,
        file_name="labeled_train.txt")
    labeled_dataset = dataset.Dataset(
        file_names=file_names,
        dataset_path=configs.dataset_path,
        labels_dict=configs.labels)
    
    for image, label, bbs in labeled_dataset:
        print(f"Image shape: {image.numpy().shape}")
        print(f"Label shape: {label.numpy().shape}")
        print(f"BBS shape: {bbs.numpy().shape}")

        print(f"Labels {label})")
        print(f"BBS {bbs}")

        break

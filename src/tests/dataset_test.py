from src import dataset
from src.utils import label_utils

import tensorflow as tf

if __name__ == "__main__":
    dataset_dir = "datasets/data/VOC2012"
    batch_size = 1
    shuffle_size = 16
    image_dims = (512, 512)
    aspect_ratios = [0.5, 1, 2]
    scales = [0, 1/3, 2/3]
    file_names = label_utils.load_data(dataset_dir)
    dataset_func = dataset.Dataset(
        dataset_dir,
        batch_size,
        shuffle_size,
        image_dims,
        aspect_ratios,
        scales)
    ds = dataset_func(file_names)

    for image, labels in ds:
        print(image.numpy().shape)
        print(labels.numpy().shape)
        break

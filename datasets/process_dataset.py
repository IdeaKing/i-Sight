# i-Sight: Process the dataset based on needed usage.
# Run Script:
"""
    cd datasets
    python process_dataset.py --dataset-dir PATH-TO-DATASET-DIR
"""
# ENSURE THAT THERE ARE ONLY TWO FOLDERS: images AND labels

import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir

    return dataset_dir


def create_training_files(dataset_dir):
    """Finds all labeled images and unlabeled images and sorts into two files."""
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    # Finds the file names of all images and labels
    images_dir_files = [file[:-4] for file in os.listdir(images_dir)]
    labels_dir_files = [file[:-4] for file in os.listdir(labels_dir)]
    # Labeled images
    labeled_images = list(set(images_dir_files).intersection(labels_dir_files))
    # Unlabeled images
    unlabeled_images = [x for x in images_dir_files if x not in labels_dir_files]
    # Write the images names to file
    labeled_image_file = os.path.join(dataset_dir, "labeled_train.txt")
    unlabeled_image_file = os.path.join(dataset_dir, "unlabeled_train.txt")
    f_labeled = open(labeled_image_file, "w")
    [f_labeled.write(x) for x in labeled_images]
    f_unlabeled = open(unlabeled_image_file, "w")
    [f_unlabeled.write(x) for x in unlabeled_images]
    f_labeled.close()
    f_unlabeled.close()


def create_student_training_files(configs):
    images_dir = os.path.join(dataset_dir, "pl_dataset")
    # Finds the file names of all images and labels
    images_dir_files = [file[:-4] for file in os.listdir(images_dir)]
    labeled_image_file = os.path.join(dataset_dir, "labeled_train.txt")

if __name__ == "__main__":
    dataset_dir = parse_args()
    print("Parsing --> ", dataset_dir)
    print("This may take a second...")
    create_training_files(dataset_dir)
    print("Completed.")
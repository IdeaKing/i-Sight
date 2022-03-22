# Adapted by Thomas Chia
# Portions of Code from https://github.com/joydeepmedhi/Anchor-Boxes-with-KMeans
# Most calculations from https://github.com/zhouyuangan/K-Means-Anchors
# More information on Anchor https://www.oreilly.com/library/view/practical-machine-learning/9781098102357/ch04.html

import csv
import argparse
import numpy as np


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    :param box: tuple or array, shifted to the origin (i. e. width and height)
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")

    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]

    iou_ = intersection / (box_area + cluster_area - intersection)
    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param clusters: numpy array of shape (k, 2) where k is the number of clusters
    :return: average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    :param boxes: numpy array of shape (r, 4)
    :return: numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    :param boxes: numpy array of shape (r, 2), where r is the number of rows
    :param k: number of clusters
    :param dist: distance function
    :return: numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    np.random.seed()
    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(
                boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters


def read_csv(path_to_csv):
    bboxes = []
    with open(path_to_csv, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            _, x1, y1, x2, y2, _ = row
            bboxes.append([float(x2) - float(x1), float(y2) - float(y1)])
    return np.array(bboxes)


def main(path_to_csv, clusters):
    bboxes = read_csv(path_to_csv)
    out = kmeans(bboxes, k=clusters)
    print("Completed KMeans Algorithn.")
    print("Accuracy: {:.2f}%".format(avg_iou(bboxes, out) * 100))
    # print(f"Boxes:\n {out}")
    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print(f"Ratios:\n {sorted(ratios)}")
    scale = np.around(out[:,  1]*np.sqrt(ratios), decimals=2).tolist()
    print(f"Scales: \n {scale}")
    return ratios, scale


if __name__ == "__main__":
    args = argparse.ArgumentParser(prog="Optimize the anchors.")
    args.add_argument("--path-to-csv", required=True)
    args.add_argument("--num-clusters", type=int, default=3)
    args = args.parse_args()

    main(args.path_to_csv, args.num_clusters)

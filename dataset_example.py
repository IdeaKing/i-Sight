# %%
# Grab the imports needed for dataset testing
import os
import tensorflow as tf
import matplotlib.pyplot as plt

from src.utils import postprocess, visualize, file_reader, label_utils
from src import dataset

# %%
# Basic configurations
DATASET_PATH = "datasets/data/VOC2012"
IMAGES_DIR = os.path.join(DATASET_PATH, "images")
LABELS_DIR = os.path.join(DATASET_PATH, "labels")
BATCHSIZE = 4
LABELS_DICT = file_reader.parse_label_file(DATASET_PATH + "/labels.txt")

# %%
file_names = dataset.load_data(dataset_path=DATASET_PATH,
                               file_name="labeled_train_.txt")
dataset_func = dataset.Dataset(file_names=file_names,
                               dataset_path=DATASET_PATH,
                               labels_dict=LABELS_DICT,
                               training_type="object_detection",
                               batch_size=BATCHSIZE,
                               shuffle_size=1,
                               images_dir="images",
                               labels_dir="labels",
                               image_dims=(512, 512),
                               augment_ds=False,
                               dataset_type="labeled")

# %%
# Take one file and read the image and labels
image, label = [], []
for x in range(BATCHSIZE):
    image_path = os.path.join(IMAGES_DIR, file_names[x] + ".jpg")
    label_path = os.path.join(LABELS_DIR, file_names[x] + ".xml")
    image.append(dataset_func.parse_process_image(image_path))
    label.append(dataset_func.parse_process_voc(label_path))

# %%
# Display the first image and print the first labels
plt.imshow(image[0])
plt.show()

print("object", label[0][0])
print("image", label[0][1])

# %%
# Show the batch of the images and labels
for x in range(BATCHSIZE):
    print("object", label[x][1])
    print("label", label[x][0])

    # Match the bounding boxes to the image
    bbox_image = visualize.draw_boxes(image[x], 
                                      labels=label[x][0], 
                                      bboxes=label[x][1], 
                                      scores=label[x][0])
    plt.imshow(bbox_image)
    plt.show()


# %%
# Run the encode the batch with anchors
# First you must convert the coordinates from xmin, ymin, xmax, ymax to 
# xcenter, ycenter, width, height

for x in range(BATCHSIZE):
    bbx = label[x][1]
    cls = label[x][0]
    print("ground truth labels", cls)
    print("ground truth bboxes", bbx)
    bbx = label_utils.to_xywh(bbx)
    out = dataset_func.encoder._encode_sample(image_shape=(512, 512),
                                              gt_boxes=tf.cast(bbx, tf.float32),
                                              classes=tf.cast(cls, tf.int32))
    print("anchored labels", out[0])
    print("anchored bboxes", out[1])
    
    # Now we convert into GT to see if anchors are right!
    labels, bboxes, scores = postprocess.FilterDetections()(
        tf.expand_dims(out[0], axis=0), 
        tf.expand_dims(out[1], axis=0))

    print("reconverted labels", labels)
    print("reconverted bboxes", bboxes)


# %%

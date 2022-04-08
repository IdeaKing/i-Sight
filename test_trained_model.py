import numpy as np
import tensorflow as tf

from src.utils.postprocess import FilterDetections
from src.utils.visualize import draw_boxes
from src.utils.file_reader import parse_label_file


def preprocess_image(image_path, image_dims):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_jpeg(
        image,
        channels=3)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(images=image,
                            size=image_dims)
    image = np.asarray(image, np.float32)    
    return image


def test(image_path, model, image_dims, label_dict, score_threshold):
    image = np.expand_dims(
        preprocess_image(image_path, image_dims),
        axis=0)  # (1, 512, 512, 3)

    pred_cls, pred_box = model(image, training=False)
    labels, bboxes, scores = FilterDetections(score_threshold)(
        labels=pred_cls,
        bboxes=pred_box)

    labels = [list(label_dict.keys())[int(l)]
              for l in labels[0]]
    bboxes = bboxes[0]
    scores = scores[0]

    image = draw_boxes(
        image=np.squeeze(image, axis=0),
        bboxes=bboxes,
        labels=labels,
        scores=scores)

    image.save("test.jpg")


if __name__ == "__main__":
    image_dims = (512, 512)
    label_dict = parse_label_file(
        path_to_label_file="datasets/data/obd_fundus/labels.txt")
    score_threshold = 0.5

    # Path to image and model
    image_path = "datasets\data\obd_fundus\images\cws_0019.jpg"
    model_path = "training_dir/fundus_d4/model-exported"
    model = tf.keras.models.load_model(model_path)

    # Test the model on the image
    test(image_path=image_path,
         model=model,
         image_dims=image_dims,
         label_dict=label_dict,
         score_threshold=score_threshold)

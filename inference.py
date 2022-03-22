import os
import argparse
import tensorflow as tf

from src.utils.postprocess import FilterDetections
from src.utils.visualize import draw_boxes
from src.utils.file_reader import parse_label_file


def preprocess_image(image_path, image_dims):
    image = tf.io.read_file(image_path)
    image = tf.io.decode_image(image, channels=3)
    image = tf.cast(image, dtype=tf.float32)/255.  # Normalize
    image = tf.image.resize(image, size=image_dims)
    image = tf.expand_dims(image, axis=0)
    return image


def test(image_path, image_dir, save_dir, model, 
         image_dims, label_dict, score_threshold, iou_threshold):
    """Preprocesses, Tests, and Postprocesses"""
    print(f"Processing {image_path}")
    image = preprocess_image(
        os.path.join(image_dir, image_path), image_dims)

    pred_cls, pred_box = model(image, training=False)
    labels, bboxes, scores = FilterDetections(
        score_threshold=score_threshold, 
        iou_threshold=iou_threshold, 
        image_dims=image_dims,
        scales=[0.67781626, 0.08602883, 0.27783391],
        aspect_ratios=[0.57, 0.59, 0.83])(
            labels=pred_cls,
            bboxes=pred_box)

    labels = [list(label_dict.keys())[int(l)]
              for l in labels[0]]
    bboxes=bboxes[0]
    scores=scores[0]

    image = draw_boxes(
        image=tf.squeeze(image, axis=0),
        bboxes=bboxes,
        labels=labels,
        scores=scores)

    image.save(os.path.join(save_dir, image_path))

if __name__ == "__main__":
    parser=argparse.ArgumentParser(
        description="Run i-Sight Tests",
        prog="i-Sight")
    parser.add_argument("--testing-image-dir",
                        type=str,
                        default="datasets/data/VOC2012/TestImages",
                        help="Path to testing images directory.")
    parser.add_argument("--save-image-dir",
                        type=str,
                        default="datasets/data/Tests",
                        help="Path to testing images directory.")
    parser.add_argument("--model-dir",
                        type=str,
                        default="training_dir/voc/model-exported",
                        help="Path to testing model directory.")
    parser.add_argument("--image-dims",
                        type=tuple,
                        default=(512, 512),
                        help="Size of the input image.")
    parser.add_argument("--labels-file",
                        type=str,
                        default="datasets/data/VOC2012/labels.txt",
                        help="Path to labels file.")
    parser.add_argument("--score-threshold",
                        type=float,
                        default=0.05,
                        help="Score threshold for NMS.")
    parser.add_argument("--iou-threshold",
                        type=float,
                        default=0.9,
                        help="IOU threshold for NMS.")
    args=parser.parse_args()

    label_dict = parse_label_file(
        path_to_label_file=args.labels_file)

    model = tf.keras.models.load_model(args.model_dir)

    if os.path.exists(args.save_image_dir) == False:
        os.mkdir(args.save_image_dir)

    for image_path in os.listdir(args.testing_image_dir):
        # Test the model on the image
        test(image_path=image_path,
             image_dir=args.testing_image_dir,
             save_dir=args.save_image_dir,
             model=model,
             image_dims=args.image_dims,
             label_dict=label_dict,
             score_threshold=args.score_threshold,
             iou_threshold=args.iou_threshold)

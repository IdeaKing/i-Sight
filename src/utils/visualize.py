# i-Sight Display Functions for Object Detection

import tensorflow as tf
from PIL import ImageDraw

from src.utils.postprocess import _image_to_pil, _parse_boxes


def draw_boxes(image,
               boxes,
               labels,
               scores,
               colors = [(0, 255, 0)]):
    """
    Draw a set of boxes formatted as [x1, y1, x2, y2] to the image `image`
    
    Parameters
    ----------
    image: ImageType
        Image where the boxes are going to be drawn
    boxes: Boxes
        Set of boxes to draw. Boxes must have the format [x1, y1, x2, y2]
    labels: Sequence[str], default None
        Classnames corresponding to boxes
    scores: FloatSequence, defalt None
    colors: Sequence[Color], default [(0, 255, 0)]
        Colors to cycle through
    Returns
    -------
    PIL.Image
    """
    image = _image_to_pil(image)
    boxes = _parse_boxes(boxes)

    """
    # Fill scores and labels with None if needed
    if labels is None:
        labels = [""] * len(boxes)
    
    if scores is None:
        scores = [""] * len(boxes)
    elif isinstance(scores, np.ndarray):
        scores = scores.reshape(-1).tolist()
    elif isinstance(scores, tf.Tensor):
        scores = scores.numpy().reshape(-1).tolist()

    # Check if scores and labels are correct
    assert len(labels) == len(boxes), \
        "Labels and boxes must have the same length"
    assert len(scores) == len(boxes), \
        "Scores and boxes must have the same length"
    """
    n_colors = len(colors)
    draw = ImageDraw.Draw(image)
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        text = str(f"{label}, {score}")
        x1, y1, x2, y2 = box
        c = colors[i % n_colors]

        draw.text([x1 + 5, y1 - 10], text)
        draw.rectangle(box, outline=c, width=2)
    
    return image

@tf.function
def printProgressBar(
    step, 
    total, 
    loss_vals):
    """Training Progress Bar"""
    decimals = 1
    length = 20
    fill = "=" #"█"
    printEnd = "\r"

    percent = ("{0:." + str(decimals) + "f}").format(100 * (step / float(total)))
    filledLength = int(length * step // total)
    bar = fill * filledLength + "-" * (length - filledLength)

    print(f"\r Step: {step}/{total} |{bar}| {percent}%" + \
          " ".join(f"loss-{i} {round(loss, 5)}" for i, loss in enumerate(loss_vals)), 
          end=printEnd)

    # Print New Line on Complete
    if iter == total: 
        print()
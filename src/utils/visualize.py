# i-Sight Display Functions for Object Detection

import tensorflow as tf
from PIL import ImageDraw, Image


def draw_boxes(image,
               bboxes,
               labels,
               scores,
               colors=[(0, 255, 0), (255, 255, 255)]):
    """
    Draw a set of boxes formatted as [x1, y1, x2, y2] to the image `image`
    """

    if isinstance(image, Image.Image):
        image = image
    elif isinstance(image, tf.Tensor):
        image = image.numpy()
    
    if image.dtype == 'float32' or image.dtype == 'float64':
        image = (image * 255.).astype('uint8')
    elif image.dtype != 'uint8':
        print(image.dtype)
        raise ValueError('Image dtype not supported')

    n_colors = len(colors)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for i, (boxes, label, score) in enumerate(zip(bboxes, labels, scores)):
        text = str(f"{label}, {score}")
        x1, y1, x2, y2 = boxes
        c = colors[i % n_colors]

        draw.text([x1 + 5, y1 - 10], text)
        draw.rectangle(boxes, outline=c, width=2)

    return image


@tf.function
def printProgressBar(
        step,
        total,
        loss_vals):
    """Training Progress Bar"""
    decimals = 1
    length = 20
    fill = "="  # "█"
    printEnd = "\r"

    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (step / float(total)))
    filledLength = int(length * step // total)
    bar = fill * filledLength + "-" * (length - filledLength)

    print(f"\r Step: {step}/{total} |{bar}| {percent}%" +
          " ".join(f"loss-{i} {round(loss, 5)}" for i,
                   loss in enumerate(loss_vals)),
          end=printEnd)

    # Print New Line on Complete
    if iter == total:
        print()

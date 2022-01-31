# Generic util functions from train.py
# Thomas Chia 12/6/2021

import os
import numpy as np
import tensorflow as tf

import src.models.efficientdet as efficientdet
import src.losses.anchor as anchor
import src.utils.nms as NMS


def object_detection_optimizer(configs):
    teacher_optimizer = tf.keras.optimizers.Adam(
        learning_rate = configs.teacher_learning_rate)
    tutor_optimizer = tf.keras.optimizers.Adam(
        learning_rate = configs.student_learning_rate)
    student_optimizer = tf.keras.optimizers.Adam()

    return teacher_optimizer, tutor_optimizer, student_optimizer


def object_detection_logits_into_labels(configs, logits):
    """Change the logits into labels for object detection."""
    anchors = anchor.Anchors(
        scales = configs.scales, 
        ratios = configs.ratios)(image_size = configs.image_dims)
    nms = NMS.NMS(configs = configs)
    box_transform = efficientdet.BoxTransform()
    clip_boxes = efficientdet.ClipBoxes(configs)

    reg_results, cls_results = logits[..., :4], logits[..., 4:]
    transformed_anchors = box_transform(anchors, reg_results)
    transformed_anchors = clip_boxes(transformed_anchors)
    scores = tf.math.reduce_max(cls_results, axis=2).numpy()
    classes = tf.math.argmax(cls_results, axis=2).numpy()
    final_boxes, final_scores, final_classes = nms(
        boxes=transformed_anchors[0, :, :],
        box_scores=np.squeeze(scores),
        box_classes=np.squeeze(classes))
    merged_output = np.concatenate(
        [final_boxes.numpy(), final_classes.numpy()],
        axis = -1)
    return merged_output, final_scores.numpy() 


def save_labels_to_xml(configs, labels, path):
    """Saves the labels created by model into xml labels."""
    # Add meta data to the xml file
    with open(path, 'w') as f:
        f.write('<annotation>\n')
        f.write('\t<folder>'+ str(configs.dataset_dir) + '</folder>\n')
        f.write('\t<filename>' + os.path.basename(path)[:-4] + ".jpg" + '</filename>\n')
        f.write('\t<path>' + path[:-4] + ".jpg" + '</path>\n')
        f.write('\t<source>\n')
        f.write('\t\t<database>AMPL-Thomas-Chia-2022</database>\n')
        f.write('\t</source>\n')
        f.write('\t<size>\n')
        f.write('\t\t<width>' + str(configs.image_dims[0]) + '</width>\n')
        f.write('\t\t<height>' + str(configs.image_dims[1]) + '</height>\n')
        f.write('\t\t<depth>3</depth>\n')
        f.write('\t</size>\n')
        f.write('\t<segmented>0</segmented>\n')
        # Loop through each of the labels and add to the xml file
        for label in labels:
            xmin, ymin, xmax, ymax, id = label
            # Convert the id number into object names
            object_name = list(configs.labels.keys()).index(id)
            # Write each coordinate and class to the file
            f.write('\t<object>\n')
            f.write('\t\t<name>' + object_name + '</name>\n')
            f.write('\t\t<pose>Unspecified</pose>\n')
            f.write('\t\t<truncated>0</truncated>\n')
            f.write('\t\t<difficult>0</difficult>\n')
            f.write('\t\t<bndbox>\n')
            f.write('\t\t\t<xmin>' + xmin + '</xmin>\n')
            f.write('\t\t\t<ymin>' + ymin + '</ymin>\n')
            f.write('\t\t\t<xmax>' + xmax + '</xmax>\n')
            f.write('\t\t\t<ymax>' + ymax + '</ymax>\n')
            f.write('\t\t</bndbox>\n')
            f.write('\t</object>\n')
        # Close the annotation tag once all the objects have been written to the file
        f.write('</annotation>\n')
        f.close() # Close the file

def read_files(file_name):
    """Reads each file line by line."""
    file_contents = []
    file = open(file_name)
    lines = file.readlines()
    for line in lines:
        file_contents.append(line.strip())
    return file_contents


def parse_label_file(path_to_label_file):
    """Parses file with labels and converts into dict. For object detection."""
    labels = open(path_to_label_file)
    label_dict = {}
    index = 0
    for label in labels:
        label_dict[str(label.strip())] = index
        index = index + 1
    return label_dict


def model_weights(model_type, configs):
    """Return directory to the model weights."""
    if model_type == "teacher":
        work_dir = configs.training_directory
        model_weights_dir = os.path.join(work_dir, "teacher")
        return model_weights_dir
    elif model_type == "student":
        work_dir = configs.training_directory
        model_weights_dir = os.path.join(work_dir, "student")
        return model_weights_dir
    else:
        work_dir = configs.training_directory
        model_weights_dir = os.path.join(work_dir, "ema")
        return model_weights_dir


def update_ema_weights(train_config, ema_model, student_model, step):
    """Update according to ema and return new weights."""
    ema_step = float(step - train_config.ema_start)
    decay = 1.0 - min(train_config.ema_decay, (ema_step + 1.0) / (ema_step + 10.0))
    decay = 1.0 if step < train_config.ema_start else decay
    new_weights = []
    for curr, new in zip(ema_model.get_weights(), student_model.get_weights()):
        new_weights.append(curr * (1 - decay) + new * decay)
    ema_model.set_weights(new_weights)


def learning_rate(
    global_step, 
    learning_rate_base, 
    total_steps, 
    num_warmup_steps=0, 
    num_wait_steps=0):
    """Get learning rate."""
    if global_step < num_wait_steps:
        return 1e-9
    global_step = global_step - num_wait_steps
    if num_warmup_steps > total_steps:
        num_warmup_steps = total_steps - 1
    rate = cosine_decay_with_warmup(
        global_step,
        learning_rate_base,
        total_steps - num_wait_steps,
        warmup_steps=num_warmup_steps,
    )
    return rate


def cosine_decay_with_warmup(
    global_step,
    learning_rate_base,
    total_steps,
    warmup_learning_rate=0.0,
    warmup_steps=0,
    hold_base_rate_steps=0,
):
    """Cosine decay schedule with warm up period.
    Cosine annealing learning rate as described in:
      Loshchilov and Hutter, SGDR: Stochastic Gradient Descent with Warm Restarts.
      ICLR 2017. https://arxiv.org/abs/1608.03983
    In this schedule, the learning rate grows linearly from warmup_learning_rate
    to learning_rate_base for warmup_steps, then transitions to a cosine decay
    schedule.
    Arguments:
        global_step {int} -- global step.
        learning_rate_base {float} -- base learning rate.
        total_steps {int} -- total number of training steps.
    Keyword Arguments:
        warmup_learning_rate {float} -- initial learning rate for warm up. (default: {0.0})
        warmup_steps {int} -- number of warmup steps. (default: {0})
        hold_base_rate_steps {int} -- Optional number of steps to hold base learning rate
                                    before decaying. (default: {0})
    Returns:
      a float representing learning rate.
    Raises:
      ValueError: if warmup_learning_rate is larger than learning_rate_base,
        or if warmup_steps is larger than total_steps.
    """

    if total_steps < warmup_steps:
        raise ValueError("total_steps must be larger or equal to " "warmup_steps.")
    learning_rate = (
        0.5
        * learning_rate_base
        * (
            1
            + np.cos(
                np.pi
                * float(global_step - warmup_steps - hold_base_rate_steps)
                / float(total_steps - warmup_steps - hold_base_rate_steps)
            )
        )
    )
    if hold_base_rate_steps > 0:
        learning_rate = np.where(
            global_step > warmup_steps + hold_base_rate_steps,
            learning_rate,
            learning_rate_base,
        )
    if warmup_steps > 0:
        if learning_rate_base < warmup_learning_rate:
            raise ValueError(
                "learning_rate_base must be larger or equal to " "warmup_learning_rate."
            )
        slope = (learning_rate_base - warmup_learning_rate) / float(warmup_steps)
        warmup_rate = slope * float(global_step) + warmup_learning_rate
        learning_rate = np.where(global_step < warmup_steps, warmup_rate, learning_rate)
    return np.where(global_step > total_steps, 0.0, learning_rate)


def update_tensorboard(
    losses, step, teacher_optimizer, tutor_optimizer):
    # Adds learning rate information to TB
    tf.summary.scalar(
        "Tutor/Learning-Rate",
        data = tutor_optimizer.learning_rate,
        step = step)
    tf.summary.scalar(
        "Teacher/Learning-Rate",
        data = teacher_optimizer.learning_rate,
        step = step)
    tf.summary.scalar(
        "Moving Dot Product: All Modes",
        data = losses.get("moving-dot-product"),
        step = step)
    tf.summary.scalar(
        "Teacher/Labeled-Data",
        data = losses.get("tutor-on-l"),
        step = step)
    tf.summary.scalar(
        "Tutor/Labeled-Data",
        data = losses.get("tutor-on-l"),
        step = step)
    tf.summary.scalar(
        "Teacher/Unlabeled-Data",
        data = losses.get("teacher-on-u"),
        step = step)
    tf.summary.scalar(
        "Tutor/Unlabeled-Data",
        data = losses.get("student-on-u"),
        step = step)

    tf.summary.flush()
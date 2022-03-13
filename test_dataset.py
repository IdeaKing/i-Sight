import tensorflow as tf

import src.dataset as dataset
import src.config as config

from src.utils.anchors import AnchorGenerator, anchor_targets_bbox

def compute_gt(images, annots, anchors, num_classes):

    labels = annots[0]
    boxes = annots[1]

    target_reg, target_clf = anchor_targets_bbox(
            anchors, images, boxes, labels, num_classes)

    return images, (target_reg, target_clf)

def _generate_anchors(anchors_config,
                      im_shape: int):

    anchors_gen = [AnchorGenerator(
            size=anchors_config.sizes[i - 3],
            aspect_ratios=anchors_config.ratios,
            stride=anchors_config.downsampling_strides[i - 3]) 
            for i in range(3, 8)]

    shapes = [im_shape // (2 ** x) for x in range(3, 8)]

    anchors = [g((size, size, 3))
               for g, size in zip(anchors_gen, shapes)]

    return tf.concat(anchors, axis=0)


if __name__=="__main__":
    configs = config.Configs(
        training_type="obd",
        dataset_path="datasets/data/VOC2012",
        training_dir="none")
    file_names = dataset.load_data(
        configs=configs,
        dataset_type="labeled")
    labeled_dataset = dataset.Dataset(
        file_names=file_names,
        configs=configs,
        dataset_type="labeled").create_dataset()

    for image, label, bbs in labeled_dataset:
        print(f"Image shape: {image.numpy().shape}")
        print(f"Label shape: {label.numpy().shape}")
        print(f"BBS shape: {bbs.numpy().shape}")

        print(f"Labels {label})")
        print(f"BBS {bbs}")

        annots = (label, bbs)

        anchors = _generate_anchors(configs, configs.image_dims[0])
        images, (target_reg, target_clf) = compute_gt(
            images=image, 
            annots=annots, 
            anchors=anchors, 
            num_classes=configs.num_classes)

        break

    

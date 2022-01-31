import numpy as np
import tensorflow as tf

import src.dataset as dataset
import src.models.model_builder as model_builder
import src.losses.losses as loss

# Prepare the GPU
tf.keras.backend.clear_session()
gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    tf.config.experimental.set_memory_growth(gpus[0], False)
tf.config.run_functions_eagerly(True)

PATH_TO_DATASET = "datasets/obd_fundus"
EPOCHS = 1

class Configs:
    # Dataset Configs
    image_dims = 512
    batch_size = 1
    training_type = "obd"
    dataset_path = PATH_TO_DATASET
    labels_dir = "labels"
    images_dir = "images"
    labels = {
        "cotton_wool_spot": 0,
        "drusen": 1,
        "geography_atrophy": 2,
        "hard_exudates": 3,
        "hemorrhages": 4,
        "optic_disc": 5,
        "macula": 6,
        "microaneurysms": 7}
    num_classes = len(labels)
    serialize_dataset = False

    # EffDet
    num_anchor_per_pixel = 9
    width_coefficient = 1.0
    depth_coefficient = 1.0
    dropout_rate = 2.0
    w_bifpn = 64
    d_bifpn = 2
    d_class = 3
    anchors = 9

    

def get_dataset():
    labeled_file_names = dataset.load_data(
        training_configs = Configs)
    labeled_dataset_maker = dataset.Dataset(
        dataset_type = "labeled",
        training_configs = Configs)
    labeled_ds = labeled_dataset_maker.create_dataset(
        labeled_file_names)
    return labeled_ds

@tf.function
def train(dataset):
    print("Building model")
    model = model_builder.build_objectdetection(
        Configs)
    print("optimizer")
    optimizer = tf.keras.optimizers.Adam(lr=0.01)

    print("running training")
    step = 0
    for epoch in range(EPOCHS):
        for images, labels in dataset:
            # print(tf.shape(images))
            with tf.GradientTape() as tape:
                # print("running model")
                logits = model(images, training=True)
                # print("calculating losses")
                losses = loss.object_detection_loss(
                    labels, logits)
            gradients = tape.gradient(losses, model.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, model.trainable_variables))
            print(step, losses)
            step += 1

if __name__ == "__main__":
    print("Getting dataset")
    dataset = get_dataset()
    print("Beginning training")
    train(dataset)
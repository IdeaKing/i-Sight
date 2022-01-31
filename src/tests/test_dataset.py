import src.dataset as dataset
import matplotlib.pyplot as plt
import tensorflow as tf

DATASET_PATH = "datasets/obd_fundus"

class Configs():
    image_dims = 512
    training_type = "obd"
    dataset_path = DATASET_PATH
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
    serialize_dataset = False

if __name__ == "__main__":
    # Prepare the GPU
    tf.keras.backend.clear_session()
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0], False)

    labeled_file_names = dataset.load_data(
        training_configs=Configs)
    
    labeled_dataset_maker = dataset.Dataset(
        dataset_type = "labeled",
        training_configs = Configs)

    labeled_ds = labeled_dataset_maker.create_dataset(labeled_file_names)

    for x, y in labeled_ds:
        plt.imshow(x.numpy())
        plt.show()
        print(y.numpy())
        break

    unlabeled_file_names = dataset.load_data(
        training_configs=Configs, 
        dataset_type = "unlabeled")

    unlabeled_dataset_maker = dataset.Dataset(
        dataset_type = "unlabeled",
        training_configs = Configs)

    unlabeled_ds = unlabeled_dataset_maker.create_dataset(unlabeled_file_names)

    for x, y in unlabeled_ds:
        plt.imshow(x.numpy())
        plt.show()
        plt.imshow(y.numpy())
        plt.show()
        break
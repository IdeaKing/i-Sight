import tensorflow as tf

from src import dataset
from src.utils import label_utils
from src.models import efficientdet
from src.losses import loss

BATCH_SIZE = 2
EPOCHS = 25
IMAGE_DIMS = (512, 512)
LABELS = label_utils.get_labels_dict()
DATASET_PATH = "datasets/data/VOC2012"
DATASET_FILES = label_utils.load_data(DATASET_PATH)
TOTAL_STEPS = len(DATASET_FILES) / BATCH_SIZE * EPOCHS
STEPS_PER_EPOCH = int(TOTAL_STEPS / EPOCHS)

def main():
    model = efficientdet.EfficientDet(num_classes=len(LABELS), )
    loss_func = loss.EffDetLoss(num_classes=len(LABELS))
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    ds = dataset.Dataset(DATASET_PATH, BATCH_SIZE)(DATASET_FILES)

    for epoch in range(EPOCHS):
        for i, (images, labels) in enumerate(ds):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss_value = loss_func(labels, logits)

            grads = tape.gradient(loss_value, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            print(f"Epoch {i + 1}/{STEPS_PER_EPOCH} Loss: {loss_value}")
        model.save_weights(f"models/efficientdet-{epoch + 1}.h5")
    tf.keras.models.save_model(model, f"models/efficientdet.h5")

if __name__ == "__main__":
    main()
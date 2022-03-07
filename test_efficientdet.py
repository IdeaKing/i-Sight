import tensorflow as tf

from src.config import Configs
from src.models.efficientdet import model_builder
from src.dataset import Dataset, load_data
from src.losses.loss import effdet_loss

tf.config.run_functions_eagerly(True)

if __name__ == "__main__":
    # Prepare the configs
    configs = Configs(
        training_type="obd",
        dataset_path="datasets/data/obd_fundus",
        training_dir="test-new-training")

    # Prepare the dataset
    files = load_data(
        configs=configs)
    labeled_dataset_builder = Dataset(
        file_names=files,
        configs=configs,
        dataset_type="labeled")
    labeled_data = labeled_dataset_builder.create_dataset()

    # Test the dataset outputs
    for x, y in labeled_data:
        print(x.numpy())
        print(y.numpy())
        break
    
    # Build the models
    model = model_builder(configs=configs, name="test-ed")
    model.summary()

    # Training configurations
    EPOCHS = 10
    STEPS_PER_EPOCH = int(len(files) / configs.batch_size)
    TOTAL_STEPS = STEPS_PER_EPOCH * EPOCHS

    loss_func = effdet_loss(configs=configs)
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-4,
        decay_steps=TOTAL_STEPS,
        decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    # Loss Metrics 
    loss_metric = tf.metrics.Mean()
    
    # Training Function
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            labels = labels.numpy()
            logits = model(images, training=True)
            loss_value = loss_func(y_true=labels, y_pred=logits)

        gradients = tape.gradient(target=loss_value, sources=model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))
        loss_metric.update_state(values=loss_value)

    for epoch in range(EPOCHS):
        i = 1
        print("Epoch: {}/{}".format(epoch, EPOCHS))
        for x, y in labeled_data:
            train_step(x, y)
            print("Step: {}/{} Loss: {}".format(i, STEPS_PER_EPOCH, loss_metric.result()))
            loss_metric.reset_states()
            i = i + 1

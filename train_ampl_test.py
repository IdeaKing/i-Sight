import tensorflow as tf

from src import train, dataset
from src.models import efficientdet
from src.losses import loss
from src.utils import file_reader

DATASET_PATH = "data/datasets/VOC2012"
UNLABELED_DATASET_FILES = "unlabeled_train.txt"
DATASET_FILES = "labeled_train.txt"
IMAGES_DIR = "images"
LABELS_DIR = "labels"
IMAGE_DIMS = (64, 64)
MODEL = "efficientdet_d0"
EPOCHS = 1
BATCH_SIZE = 1
UNLABELED_BATCH_SIZE = 1
LABELS_DICT = file_reader.parse_label_file(DATASET_PATH + "/labels.txt")
NUM_CLASSES = len(labels_dict)
TRAINING_TYPE = "object_detection"

if __name__ == "__main__":
    unlabeled_file_names = dataset.load_data(dataset_path=DATASET_PATH,
                                             file_name=UNLABELED_DATASET_FILES)
    unlabeled_dataset = dataset.Dataset(file_names=unlabeled_file_names,
                                        dataset_path=DATASET_PATH,
                                        labels_dict=LABELS_DICT,
                                        training_type=TRAINING_TYPE,
                                        batch_size=UNLABELED_BATCH_SIZE,
                                        shuffle_size=1,
                                        images_dir=IMAGES_DIR,
                                        labels_dir=LABELS_DIR,
                                        image_dims=IMAGE_DIMS,
                                        augment_ds=False,
                                        dataset_type="labeled")()

    file_names = dataset.load_data(dataset_path=DATASET_PATH,
                                    file_name=DATASET_FILES)
    labeled_dataset = dataset.Dataset(file_names=file_names,
                                      dataset_path=DATASET_PATH,
                                      labels_dict=LABELS_DICT,
                                      training_type=TRAINING_TYPE,
                                      unlabeled_batch_size=UNLABELED_BATCH_SIZE,
                                      shuffle_size=1,
                                      images_dir=IMAGES_DIR,
                                      labels_dir=LABELS_DIR,
                                      image_dims=IMAGE_DIMS,
                                      augment_ds=False,
                                      dataset_type="labeled")()

    total_steps = int((len(file_names) / BATCH_SIZE) * EPOCHS)

    # Models
    teacher_model = efficientdet.get_efficientdet(name=MODEL,
                                                    input_shape=IMAGE_DIMS,
                                                    num_classes=NUM_CLASSES,
                                                    from_pretrained=True)
    tutor_model = efficientdet.get_efficientdet(name=MODEL,
                                                input_shape=IMAGE_DIMS,
                                                num_classes=NUM_CLASSES,
                                                from_pretrained=True)
    ema_model = efficientdet.get_efficientdet(name=MODEL,
                                                input_shape=IMAGE_DIMS,
                                                num_classes=NUM_CLASSES,
                                                from_pretrained=True)
    models = [teacher_model, tutor_model, ema_model]

    # Optimizers
    teacher_optimizer = tf.keras.optimizers.Adam(learning_rate=0.005)
    tutor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    optimizers = [teacher_optimizer, tutor_optimizer]

    # Loss func
    loss_func = loss.EffDetLoss(num_classes=NUM_CLASSES, include_iou=None)

    # Create the training function
    train_func = train.Train(training_dir=args.training_dir,
                                epochs=args.epochs,
                                total_steps=total_steps,
                                input_shape=args.image_dims,
                                precision=args.precision,
                                training_type=args.training_type,
                                max_checkpoints=args.max_checkpoints,
                                checkpoint_frequency=args.checkpoint_frequency,
                                save_model_frequency=args.save_model_frequency,
                                print_loss=args.print_loss,
                                log_every_step=args.log_every_step,
                                from_checkpoint=args.from_checkpoint)

    learning_rates = {"teacher_learning_rate": train_func.teacher_learning_rate,
                      "teacher_learning_rate_warmup": train_func.teacher_learning_rate_warmup,
                      "teacher_learning_rate_numwait": train_func.teacher_learning_rate_numwait,
                      "tutor_learning_rate": train_func.tutor_learning_rate,
                      "tutor_learning_rate_warmup": train_func.tutor_learning_rate_warmup,
                      "tutor_learning_rate_numwait": train_func.tutor_learning_rate_numwait}
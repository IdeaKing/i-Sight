import src.config as config
import src.dataset as dataset
import src.train as train
import src.utils.arg_parser as arg_parser

import tensorflow as tf

tf.keras.backend.clear_session()
tf.config.run_functions_eagerly(True)

def main(args):
    configs = config.Configs(
        training_type=args.training_type,
        dataset_path=args.dataset_path,
        training_dir=args.training_dir)
    if args.mixed_precision == True:
        configs.mixed_precision == True
        tf.keras.mixed_precision.set_global_policy(
            "mixed_float16")
    labeled_training_files = dataset.load_data(
        configs,
        dataset_type="labeled")
    unlabeled_training_files = dataset.load_data(
        configs,
        dataset_type="unlabeled")
    # Update the configs after finding all files
    configs.update_training_configs(
        dataset_size=len(labeled_training_files),
        unlabeled_data_size=len(unlabeled_training_files))
    l_dataset = dataset.Dataset(
        file_names=labeled_training_files,
        configs=configs,
        dataset_type="labeled").create_dataset()
    u_dataset = dataset.Dataset(
        file_names=unlabeled_training_files,
        configs=configs,
        dataset_type="unlabeled").create_dataset()
    train.train_ampl(
        configs=configs,
        lb_dataset=l_dataset,
        ul_dataset=u_dataset)

if __name__ == "__main__":
    args = arg_parser.args
    main(args)
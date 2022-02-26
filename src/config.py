# Thomas Chia i-Sight Configs


import os
import src.utils.training_utils as t_utils


class Configs():
    def __init__(self, training_type, dataset_path, training_dir):
        # Arg Parsed
        self.training_type = training_type
        self.mixed_precision = False
        self.dataset_path = dataset_path
        self.training_dir = training_dir
        self.serialized_dataset = False
        self.finetune = True
        # Directory 
        self.model_save = os.path.join(
            training_dir, 
            "exported-models")
        self.tensorboard_dir = os.path.join(
            training_dir,
            "tensorboard")
        self.serialized_dir = os.path.join(
            dataset_path,
            "serialized")
        self.labels_dir = "labels"
        self.images_dir = "images"
        # Dataset
        self.batch_size = 2 # Keep in multiples of 8 # same as mpl batch size
        self.unlabeled_batch_size = None # same as uda batch size
        self.buffer_size = 64
        self.shuffle_size = 64
        self.dataset_size = None # Used to compute steps
        # Training
        self.transfer_learning = "imagenet" # Or name of directory
        self.epochs = 150
        self.total_steps = None # To be updated with dataset
        self.evaluate = 5 # Evaluates every 5 epochs
        self.max_checkpoints = 5
        # Student Training
        self.student_epochs = 50
        self.student_total_steps = None
        self.student_lr = 0.00004
        self.student_batch_size = self.batch_size * 2
        # MPL
        # self.mpl_num_augments = 2
        # self.mpl_augment_magnitude = 16
        # self.mpl_augment_cutout_const = 32 // 8
        # self.mpl_augment_translate_const = 32 // 8
        # self.mpl_augment_ops_path = "dataset/augmentation.txt"
        # self.mpl_batch_size = None
        # self.mpl_unlabeled_batch_size_multiplier = 7
        self.mpl_label_smoothing = 0.15
        self.uda_label_temperature = 0.7
        self.uda_threshold = 0.5
        self.uda_weight = 8.0
        self.uda_steps = 5000
        self.ema_decay = 0.995
        self.ema_start = 0
        self.tutor_learning_rate = 0.05
        self.tutor_learning_rate_warmup = 50
        self.tutor_learning_rate_numwait = 50
        self.teacher_learning_rate = 0.08
        self.teacher_learning_rate_warmup = 50
        self.teacher_learning_rate_numwait = 0

        if self.training_type == "cls" or self.training_type == "":
            # Input 
            self.image_dims = (473, 473)
            # Dir info
            self.labels_path = os.path.join(
                dataset_path, "labels.txt")
            self.labels = t_utils.read_files(self.labels_path)
            self.num_classes = len(self.labels)
            # Model
            self.student_dropout_rate = 0.2
            self.teacher_dropout_rate = 0.2
            # Optimizer
            self.student_optimizer_momentum = 0.9
            self.student_optimizer_nesterov = True
            self.mpl_optimizer_momentum = 0.9
            self.mpl_optimizer_nesterov = True
            self.mpl_optimizer_grad_bound = 1e9

        elif self.training_type == "obd":
            self.network_type = "D0"
            # Network configurations
            self.image_size = {"D0": (512, 512), "D1": (640, 640), 
                               "D2": (768, 768), "D3": (896, 896), 
                               "D4": (1024, 1024), "D5": (1280, 1280), 
                               "D6": (1408, 1408), "D7": (1536, 1536)}
            self.width_coefficient = {"D0": 1.0, "D1": 1.0, 
                                      "D2": 1.1, "D3": 1.2, 
                                      "D4": 1.4, "D5": 1.6, 
                                      "D6": 1.8, "D7": 1.8}
            self.depth_coefficient = {"D0": 1.0, "D1": 1.1, 
                                      "D2": 1.2, "D3": 1.4, 
                                      "D4": 1.8, "D5": 2.2, 
                                      "D6": 2.6, "D7": 2.6}
            self.dropout_rate = {"D0": 0.2, "D1": 0.2, 
                                "D2": 0.3, "D3": 0.3, 
                                "D4": 0.4, "D5": 0.4, 
                                "D6": 0.5, "D7": 0.5}
            self.w_bifpn = {"D0": 64, "D1": 88, 
                            "D2": 112, "D3": 160, 
                            "D4": 224, "D5": 288, 
                            "D6": 384, "D7": 384}
            self.d_bifpn = {"D0": 2, "D1": 3, 
                            "D2": 4, "D3": 5, 
                            "D4": 6, "D5": 7, 
                            "D6": 8, "D7": 8}
            self.d_class = {"D0": 3, "D1": 3, 
                            "D2": 3, "D3": 4, 
                            "D4": 4, "D5": 4, 
                            "D6": 5, "D7": 5}
            # Update the Params
            self.image_dims = self.image_size[self.network_type]
            self.width_coefficient = self.width_coefficient[self.network_type]
            self.depth_coefficient = self.depth_coefficient[self.network_type]
            self.dropout_rate = self.dropout_rate[self.network_type]
            self.w_bifpn = self.w_bifpn[self.network_type]
            self.d_bifpn = self.d_bifpn[self.network_type]
            self.d_class = self.d_class[self.network_type]
            # Loss functions
            self.alpha = 0.25
            self.gamma = 2.00
            # Dataset Parsing
            self.labels_path = os.path.join(
                dataset_path, "labels.txt")
            self.labels = t_utils.parse_label_file(self.labels_path)
            self.num_classes = len(self.labels)
            # Post processing
            self.score_threshold = 0.01
            self.iou_threshold = 0.5
            # Anchor processng
            self.max_box_num = 200
            self.anchors = 9
            self.ratios = [0.5, 1, 2]
            self.scales = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]
            self.downsampling_strides = [8, 16, 32, 64, 128]
            self.sizes = [32, 64, 128, 256, 512]

        elif self.training_type == "seg":
            # Input 
            self.image_dims = (473, 473)
            # Dir info
            self.labels_path = os.path.join(
                dataset_path, "labels.txt")
            self.labels = t_utils.parse_label_file(self.labels_path)
            self.num_classes = len(self.labels)


    def update_training_configs(self, dataset_size, unlabeled_data_size):
        """Updates training configs related to steps and warmup."""
        self.total_steps = int(
            dataset_size / self.batch_size) * self.epochs
        # self.finetune_total_steps = int(
        #     dataset_size / self.finetune_batch_size) * self.finetune_epochs
        self.unlabeled_batch_size = int(
            unlabeled_data_size / (dataset_size / self.batch_size))
        self.mpl_batch_size = self.batch_size
        self.student_learning_rate_warmup = int(
            0.05 * self.total_steps)
        self.teacher_learning_rate_warmup = int(
            0.0 * self.total_steps)
        self.student_learning_rate_numwait = int(
            0.02 * self.total_steps)
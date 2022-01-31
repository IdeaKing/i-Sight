# i-Sight Advanced Meta Pseudo Labels Training Loop
# Thomas Chia

import tensorflow as tf

import src.models.model_builder as model_builder
import src.training_utils as t_utils

def train_ampl(training_configs, l_dataset, u_dataset):

    if training_configs.training_type=="obd":
        teacher_model = model_builder.build_objectdetection(
            training_configs)
        student_model = model_builder.build_objectdetection(
            training_configs)
        ema_model = model_builder.build_objectdetection(
            training_configs)
    elif training_configs.training_type=="cls":
        None
    elif training_configs.training_type=="seg":
        None

    
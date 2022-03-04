# i-Sight Advanced Meta Pseudo Labels Training Loop
# Thomas Chia

import os
import logging
import shutil

import tensorflow as tf

import src.models.efficientdet as efficientdet
import src.utils.training_utils as t_utils
import src.losses.effdet_loss as effdet_loss
import src.losses.uda_loss as uda_loss

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

def train_ampl(configs, lb_dataset, ul_dataset):
    """ The training pipeline for AMPL.
    :params configs: Class containing configs.
    :params l_dataset (tf.data.Dataset): Labeled Dataset
    :params u_dataset (tf.data.Dataset): Unlabeled Dataset
    :returns: Trained model
    """
    # Deletes the old directory if not continuing training
    if ((configs.transfer_learning is not True or
        configs.transfer_learning == "imagenet") and
        os.path.exists(configs.training_dir)):
        input("Press Enter to delete the current directory and continue.")
        shutil.rmtree(configs.training_dir)
    # Makes the training directory if it does not exist
    if not os.path.exists(configs.training_dir):
        os.makedirs(configs.training_dir)
    # Sets up the logging files module
    logging.basicConfig(
        level=logging.DEBUG,
        filename=os.path.join(configs.training_dir, "training.log"),
        filemode="w")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)
    # Initialize Tensorboard
    tensorboard_dir = os.path.join(
        configs.training_dir, "tensorboard")
    if os.path.exists(tensorboard_dir) == False:
        os.makedirs(tensorboard_dir)
    tensorboard_file_writer = tf.summary.create_file_writer(tensorboard_dir)
    tensorboard_file_writer.set_as_default()

    # Define Loss Functions and Training Configs
    if configs.training_type=="obd":
        uda_func = uda_loss.UDA(configs=configs)
        loss_func = effdet_loss.FocalLoss(configs=configs)
        object_detection_pseudo_labels = t_utils.PseudoLabelObjectDetection(
            configs)
        teacher_optimizer, tutor_optimizer, _ = t_utils.object_detection_optimizer(
            configs)
        # Define the models for training
        teacher_model = efficientdet.model_builder(
            configs=configs,
            name="teacher")
        tutor_model = efficientdet.model_builder(
            configs,
            name="tutor")
        ema_model = efficientdet.model_builder(
            configs,
            name="ema")
        logging.info("Created object detection models.")

    # Define the checkpoint directories
    teacher_checkpoint_dir = os.path.join(
        configs.training_dir, "teacher")
    tutor_checkpoint_dir = os.path.join(
        configs.training_dir, "tutor")
    ema_checkpoint_dir = os.path.join(
        configs.training_dir, "ema")
    # Define the full model directories
    tutor_exported_dir = os.path.join(
        configs.training_dir, "tutor-exported")
    ema_exported_dir = os.path.join(
        configs.training_dir, "ema-exported")
    # Checkpoints
    teacher_checkpoint = tf.train.Checkpoint(
        optimizer=teacher_optimizer,
        model=teacher_model)
    teacher_checkpoint_manager = tf.train.CheckpointManager(
        teacher_checkpoint,
        teacher_checkpoint_dir,
        configs.max_checkpoints)
    tutor_checkpoint = tf.train.Checkpoint(
        optimizer=tutor_optimizer, 
        model=tutor_model)
    tutor_checkpoint_manager = tf.train.CheckpointManager(
        tutor_checkpoint,
        tutor_checkpoint_dir,
        configs.max_checkpoints)
    ema_checkpoint = tf.train.Checkpoint(
        model=ema_model)
    ema_checkpoint_manager = tf.train.CheckpointManager(
        ema_checkpoint,
        ema_checkpoint_dir,
        configs.max_checkpoints)

    # Run training for the specific training type
    if configs.training_type=="obd":
        @tf.function
        def train_step(images, labels, step):
            """Trains one step of AMPL.
            :params images (dict): Contains "all", "unlabeled", and "labeled" images
            :params labels: Array of bounding box annotations
            :returns: A dict of losses.
            """
            # Step 1: Train on teacher
            with tf.GradientTape() as te_tape:
                all_logits = teacher_model(
                    images["all"],
                    training=True)
                logits, labels, masks, loss = uda_func(
                    y_true=labels,
                    y_pred=tf.cast(all_logits, tf.float32))
 
            # Only run AMPL after a certain number of steps
            # This is to prevent the 0-labels problem
            if step > configs.warmup_steps:
                # Change teacher outputs into pseudo-labels
                labels["u_aug"] = object_detection_pseudo_labels(
                    logits=logits["u_aug"])
            
                # Step 2 Run on Tutor
                with tf.GradientTape() as tu_tape:
                    logits["tu_on_u_aug_and_l"] = tutor_model(
                        images["u"],
                        training=True)
                    logit_tu_on_u, logit_tu_on_l = tf.split(
                        logits["tu_on_u_aug_and_l"],
                        [configs.unlabeled_batch_size,
                        configs.batch_size],
                        axis=0)
                    # Loss between teacher and student
                    logits["tu_on_u"] = logit_tu_on_u
                    logits["tu_on_l_old"] = logit_tu_on_l
                    # print("tutor unlabeled")
                    loss["tu_on_u"] = loss_func(
                        y_true=labels["u_aug"],
                        y_pred=tf.cast(logits["tu_on_u"], tf.float32))
                    # Loss on labeled data
                    # print("tutor labeled")
                    loss["tu_on_l_old"] = loss_func(
                        y_true=labels["l"],
                        y_pred=tf.cast(logits["tu_on_l_old"], tf.float32))
                    if configs.mixed_precision is True:
                        loss["s_on_u"] = tutor_optimizer.get_scaled_loss(
                            loss["s_on_u"])
                try:
                    # print("Working Tutor on Unlabeled {}".format(loss["tu_on_u"]))
                    tutor_grad_unlabeled = tu_tape.gradient(
                        loss["tu_on_u"],
                        tutor_model.trainable_variables)
                    tutor_grad_unlabeled, _ = tf.clip_by_global_norm(
                        tutor_grad_unlabeled, 
                        configs.mpl_optimizer_grad_bound)
                    if configs.mixed_precision is True:
                        student_grad_unlabeled = tutor_optimizer.get_unscaled_gradients(
                            student_grad_unlabeled)
                    tutor_optimizer.apply_gradients(
                        zip(tutor_grad_unlabeled, 
                        tutor_model.trainable_variables))
                except:
                    print("----------- Broken Tutor Gradient. -----------")
                    print("Tutor on Unlabeled {}".format(loss["tu_on_u"]))
                    print("Labels on Augmentened {}".format(labels["u_aug"]))
                    print("shape of labels {}".format(labels["u_aug"].shape))
                    exit()

                # Step 3 Student on labeled values + dot product calculation
                # print("tutor label new")
                logits["tu_on_l_new"] = tutor_model(images["l"])
                loss["tu_on_l_new"] = loss_func(
                    y_true=labels["l"],
                    y_pred=tf.cast(logits["tu_on_l_new"], tf.float32)) # / float(configs.unlabeled_batch_size)
                dot_product = loss["tu_on_l_new"] - loss["tu_on_l_old"]
                limit = 3.0**(0.5)
                moving_dot_product = tf.random_uniform_initializer(
                    minval=-limit, maxval=limit)(shape=dot_product.shape)
                moving_dot_product = tf.Variable(
                    initial_value=moving_dot_product,
                    trainable=False,
                    dtype = tf.float32)
                dot_product = dot_product - moving_dot_product
                dot_product = tf.stop_gradient(dot_product)

                # Step 4: Optimize the teacher on teacher and student performance
                with te_tape:
                    loss["mpl"] = loss_func(
                        y_true=labels["u_aug"],
                        y_pred=tf.cast(logits["u_aug"], tf.float32)) # / float(configs.unlabeled_batch_size)
                    uda_weight = configs.uda_weight * tf.math.minimum(
                            1., tf.cast(configs.total_steps, 
                                tf.float32) / \
                            float(configs.uda_steps))
                    loss["teacher"] = tf.reduce_sum(
                        loss["u"] * uda_weight + \
                        loss["l"] + \
                        loss["mpl"] * dot_product)
                    if configs.mixed_precision is True:
                        teacher_loss = teacher_optimizer.get_scaled_loss(
                            teacher_loss)
                teacher_grad = te_tape.gradient(
                    loss["teacher"], teacher_model.trainable_variables)
                teacher_grad, _ = tf.clip_by_global_norm(
                    teacher_grad, configs.mpl_optimizer_grad_bound)
                if configs.mixed_precision is True:
                    teacher_grad = teacher_optimizer.get_unscaled_gradients(
                        teacher_grad)
                teacher_optimizer.apply_gradients(
                    zip(teacher_grad, teacher_model.trainable_variables))
                logging.info("Step-{} L-Loss: {} Teacher-Loss: {} Tutor-L-Loss: {}".format(
                    step, loss["l"], loss["teacher"], loss["tu_on_l_new"]))
                return loss
            else:
                loss["teacher"] = loss["l"] 
                teacher_grad = te_tape.gradient(
                    loss["teacher"], teacher_model.trainable_variables)
                teacher_grad, _ = tf.clip_by_global_norm(
                    teacher_grad, configs.mpl_optimizer_grad_bound)
                if configs.mixed_precision is True:
                    teacher_grad = teacher_optimizer.get_unscaled_gradients(
                        teacher_grad)
                teacher_optimizer.apply_gradients(
                    zip(teacher_grad, teacher_model.trainable_variables))
                logging.info("Step-{} Teacher-Loss: {}".format(
                    step, loss["teacher"]))
                return loss

        # The training loop
        global_step = 0
        for epoch in range(configs.epochs):
            logging.info("Epoch {}".format(epoch))
            for step, (lb_image, lb_label) in enumerate(lb_dataset):
                # Grab data from unlabeled dataset
                lb_label = lb_label.numpy()
                or_image, au_image = next(iter(ul_dataset))
                # Group the data into dicts for easy access during training
                images = {
                    "all": tf.concat([lb_image, or_image, au_image], axis=0),
                    "u": tf.concat([au_image, lb_image], axis=0),
                    "l": lb_image
                    }
                labels = {"l": lb_label}
                # Run one step of training
                losses = train_step(images, labels, global_step)
                t_utils.update_ema_weights(
                    configs,
                    ema_model,
                    tutor_model,
                    step)
                # Update learning rate
                tutor_optimizer.learning_rate.assign(
                    t_utils.learning_rate(
                        global_step,
                        configs.tutor_learning_rate,
                        configs.total_steps,
                        configs.tutor_learning_rate_warmup,
                        configs.tutor_learning_rate_numwait))
                teacher_optimizer.learning_rate.assign(
                    t_utils.learning_rate(
                        global_step,
                        configs.teacher_learning_rate,
                        configs.total_steps,
                        configs.teacher_learning_rate_warmup,
                        configs.teacher_learning_rate_numwait))
                global_step += 1
                # Save checkpoints if needed
                if step % configs.checkpoint_frequency == 0:
                    teacher_checkpoint_manager.save()
                    if step > configs.warmup_steps:
                        tutor_checkpoint_manager.save()
                        ema_checkpoint_manager.save()
                # Tensorboard update
                if step % configs.tensorboard_log == 0:
                    tf.summary.scalar(
                        "Teacher/Learning-Rate",
                        data=teacher_optimizer.learning_rate,
                        step=global_step)
                    tf.summary.scalar(
                        "Tutor/Learning-Rate",
                        data=tutor_optimizer.learning_rate,
                        step=global_step)
                    t_utils.update_tensorboard(
                        losses=losses,
                        step=global_step)
            # Saving the model
            if global_step % configs.model_frequency == 0:
                tf.keras.models.save_model(
                    tutor_model, tutor_exported_dir)
                if step > configs.warmup_steps:
                    tf.keras.models.save_model(
                        ema_model, ema_exported_dir)

    elif configs.training_type == "cls":
        None
    elif configs.training_type == "seg":
        None

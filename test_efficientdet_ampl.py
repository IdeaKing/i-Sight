# Test the AMPL Training Pipeline
# Thomas Chia

import os
import shutil
import numpy as np
import tensorflow as tf

import src.dataset as dataset
import src.config as config
import src.losses.uda_loss as uda_loss
import src.losses.effdet_loss as effdet_loss
import src.models.efficientdet as efficientdet
import src.utils.training_utils as training_utils

def train_mpl(l_dataset, u_dataset, configs):
    """Trains the MPL Models."""
    # Deletes the old directory if not continuing training
    if ((configs.transfer_learning != True or 
        configs.transfer_learning == "imagenet") and 
        os.path.exists(configs.training_dir)):

        shutil.rmtree(configs.training_dir)
    # Makes the training directory if it does not exist
    if not os.path.exists(configs.training_dir):
        os.makedirs(configs.training_dir)

    # Initialize Tensorboard
    tensorboard_dir = os.path.join(
        configs.training_dir, "tensorboard")
    if os.path.exists(tensorboard_dir) == False:
        os.makedirs(tensorboard_dir)
    tensorboard_file_writer = tf.summary.create_file_writer(tensorboard_dir)
    tensorboard_file_writer.set_as_default()

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
    
    # Build the models
    teacher_model = efficientdet.build_model(
        configs, 
        name = "teacher")
    tutor_model = efficientdet.build_model(
        configs, 
        name = "tutor")
    ema_model = efficientdet.build_model(
        configs, 
        name = "ema")
    
    # Create the optimizers
    optimizers = training_utils.object_detection_optimizer()
    teacher_optimizer, tutor_optimizer, _ = optimizers

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
    
    # Training steps
    best_val_loss = 0
    step = tf.Variable(0, dtype=tf.int64)
    continue_step = 0
    total_steps = configs.total_steps

    # Training losses
    focal_func = effdet_loss.FocalLoss(configs = configs)
    regression_func = effdet_loss.RegressionLoss(configs = configs)
    uda_func = uda_loss.UDA(configs = configs)

    @tf.function
    def train_step(
        step,
        l_images,
        l_labels,
        u_orgims,
        u_augims,
        uda_weight):
        """Train one step on the data."""
        l_labels = l_labels.numpy()
        all_images = tf.concat(
            [l_images, u_orgims, u_augims], 
            axis = 0)
        # Step 1 Run on Teacher
        with tf.GradientTape() as te_tape:
            all_logits = teacher_model(all_images, training = True)
            logits, labels, masks, loss = uda_func(
                l_labels, 
                tf.cast(all_logits, tf.float32))
        # Process the outputs of the teacher so that the loss can be computed
        # Apply NMS etc.
        logits["u_aug"] = training_utils.object_detection_logits_into_labels(
            configs = configs, 
            logits = logits["u_aug"])

        # Step 2 Run on Tutor
        u_aug_and_l_images = tf.concat(
            [u_augims, l_images], 
            axis = 0)
        with tf.GradientTape() as tu_tape:
            logits["tu_on_u_aug_and_l"] = tutor_model(
                u_aug_and_l_images, training=True)
            # Split and calculate for labeled and unlabeled
            logit_tu_on_u, logit_tu_on_l = tf.split(
                logits["tu_on_u_aug_and_l"],
                [u_augims.shape, 
                 l_images.shape],
                axis = 0)
            logits["tu_on_u"] = logit_tu_on_u
            logits["tu_on_l_old"] = logit_tu_on_l
            # Calculate the loss
            loss["tu_on_u"] = tf.math.reduce_mean(
                focal_func(
                    y_true = tf.cast(
                        tf.stop_gradient(
                            tf.nn.sigmoid(logits["u_aug"])),
                        tf.float32),
                    y_pred = tf.cast(
                        logits["tu_on_u"], tf.float32))) + \
                              tf.math.reduce_mean(
                regression_func(
                    y_true = tf.cast(
                        tf.stop_gradient(
                            tf.nn.sigmoid(logits["u_aug"])),
                        tf.float32),
                    y_pred = tf.cast(
                        logits["tu_on_u"], tf.float32)))
            # Loss of Labeled Data
            loss["tu_on_l_old"] = tf.reduce_mean(
                focal_func(
                    y_true = labels["l"],
                    y_pred = logits["tu_on_l_old"])) + \
                                tf.reduce_mean(
                regression_func(
                    y_true = labels["l"],
                    y_pred = logits["tu_on_l_old"])) \
                / float(configs.unlabeled_batch_size)
            if configs.mixed_precision:
                loss["s_on_u"] = tutor_optimizer.get_scaled_loss(
                    loss["s_on_u"])
        student_grad_unlabeled = tu_tape.gradient(
            loss["s_on_u"], 
            tutor_model.trainable_variables)
        student_grad_unlabeled, _ = tf.clip_by_global_norm(
            student_grad_unlabeled, 
            configs.mpl_optimizer_grad_bound)
        if configs.mixed_precision:
            student_grad_unlabeled = tutor_optimizer.get_unscaled_gradients(
                student_grad_unlabeled)
        tutor_optimizer.apply_gradients(
            zip(student_grad_unlabeled, 
            tutor_model.trainable_variables))

        # Step 3 Student on labeled values + dot product calculation
        logits["tu_on_l_new"] = tutor_model(l_images)
        loss["tu_on_l_new"] = tf.math.reduce_mean(
                focal_func(
                    y_true = labels["l"],
                    y_pred = logits["tu_on_l_new"])) + \
                              tf.math.reduce_mean(
                regression_func(
                    y_true = labels["l"],
                    y_pred = logits["tu_on_l_new"])) / \
                float(configs.unlabeled_batch_size)
        dot_product = loss["tu_on_l_new"] - loss["tu_on_l_old"]    
        limit = 3.0**(0.5)
        moving_dot_product = tf.random_uniform_initializer(
            minval=-limit, maxval=limit)(shape=dot_product.shape)
        moving_dot_product = tf.Variable(
            initial_value = moving_dot_product,
            trainable = False,
            dtype = tf.float32)
        dot_product = dot_product - moving_dot_product
        dot_product = tf.stop_gradient(dot_product)

        # Step 4: Calculate teacher loss on student performance
        with te_tape:
            loss["mpl"] = tf.math.reduce_mean(
                focal_func(
                    y_true = tf.cast(
                        tf.stop_gradient(
                            tf.nn.sigmoid(logits["u_aug"])),
                        tf.float32),
                    y_pred = tf.cast(
                        logits["u_aug"], tf.float32))) + \
                              tf.math.reduce_mean(
                regression_func(
                    y_true = tf.cast(
                        tf.stop_gradient(
                            tf.nn.sigmoid(logits["u_aug"])),
                        tf.float32),
                    y_pred = tf.cast(
                        logits["u_aug"], tf.float32))) / \
                tf.convert_to_tensor(
                    configs.unlabeled_batch_size, 
                    dtype=tf.float32)
            uda_weight = configs.uda_weight * tf.math.minimum(
                    1., tf.cast(configs.total_steps, 
                        tf.float32) / \
                    float(configs.uda_steps))
            teacher_loss = tf.reduce_sum(
                loss["u"] * uda_weight + \
                loss["l"] + \
                loss["mpl"] * dot_product)
            # Scale the the teacher loss.
            if configs.mixed_precision:
                teacher_loss = teacher_optimizer.get_scaled_loss(
                    teacher_loss)
        teacher_grad = te_tape.gradient(
            teacher_loss, teacher_model.trainable_variables)
        teacher_grad, _ = tf.clip_by_global_norm(
            teacher_grad, configs.mpl_optimizer_grad_bound)
        if configs.mixed_precision:
            teacher_grad = teacher_optimizer.get_unscaled_gradients(
                teacher_grad)
        teacher_optimizer.apply_gradients(
            zip(teacher_grad, teacher_model.trainable_variables))
        
        tf.print(
            "Step: ", step,
            "dot-product", dot_product,
            "moving-dot-product", moving_dot_product,
            "teacher-on-l", loss["l"],
            "teacher-on-u", loss["u"],
            "tutor-on-u", loss["tu_on_u"],
            "tutor-on-l", loss["tu_on_l_new"])

        return {
            "dot-product": dot_product,
            "moving-dot-product": moving_dot_product,
            "teacher-on-l": loss["l"],
            "teacher-on-u": loss["u"],
            "tutor-on-u": loss["tu_on_u"],
            "tutor-on-l": loss["tu_on_l_new"]}
    
    for epoch in range(configs.epochs):
        print(" <---------- Epoch: " + str(epoch) + " ---------->")
        for l_image, label, bboxs in l_dataset:
            u_image, u_augim = next(iter(u_dataset))
            step = step + 1
            tutor_optimizer.learning_rate.assign(
                training_utils.learning_rate(
                    step,
                    configs.student_learning_rate,
                    total_steps,
                    configs.student_learning_rate_warmup,
                    configs.student_learning_rate_numwait))            
            teacher_optimizer.learning_rate.assign(
                training_utils.learning_rate(
                    step,
                    configs.teacher_learning_rate,
                    total_steps,
                    configs.teacher_learning_rate_warmup,
                    configs.teacher_learning_rate_numwait))

            uda_weight = configs.uda_weight * tf.math.minimum(
                1.0, float(step) / float(configs.uda_steps))
            # tf.summary.trace_on(graph = True)
            losses = train_step(
                step, l_image, [label, bboxs], u_image, u_augim, uda_weight)
            training_utils.update_ema_weights(
                configs, ema_model, tutor_model, step)
            training_utils.update_tensorboard(
                losses = losses, 
                step = step, 
                teacher_optimizer = teacher_optimizer, 
                tutor_optimizer = tutor_optimizer,
                tensorboard_dir = tensorboard_dir)
            # Save each checkpoint, only the best
            teacher_checkpoint_manager.save()
            tutor_checkpoint_manager.save()
            ema_checkpoint_manager.save()
        
        # Save model every epoch
        tf.keras.models.save_model(
            tutor_model, tutor_exported_dir)
        tf.keras.models.save_model(
            ema_model, ema_exported_dir)

def train_pl(l_dataset, u_dataset, configs):
    """Trains the PL Student."""
    # Clear backend from MPL training
    tf.keras.backend.clear_session()
    # Load the tutor model, and save the predictions.
    tutor_exported_dir = os.path.join(
        configs.training_dir, 
        "tutor-exported")
    tutor_model = tf.keras.models.load_model(
        tutor_exported_dir)
    pl_dataset_dir = os.path.join(
        configs.dataset_path,
        "pl_dataset")
    # Deletes the old directory for pl dataset
    if os.path.exists(pl_dataset_dir):
        shutil.rmtree(pl_dataset_dir)
    # Makes the training directory if it does not exist
    if not os.path.exists(pl_dataset_dir):
        os.makedirs(pl_dataset_dir)

    # Step 5: Run tutor model on unlabeled data and save.
    counter = 0
    for u_orgim, _ in u_dataset:
        logits = tutor_model(u_orgim, training = False)
        labels, _ = training_utils.object_detection_logits_into_labels(
            configs = configs,
            logits = logits)
        for i in range(labels.shape[0]):
            counter+=1
            file_name = configs.training_type + "-" + str(counter).zfill(6)
            file_path_img = os.path.join(
                str(pl_dataset_dir)+"image",
                file_name + ".jpg")
            file_path_xml = os.path.join(
                str(pl_dataset_dir)+"label",
                file_name + ".xml")
            # Save the image
            tf.keras.utils.save_img(
                path = file_path_img,
                x = np.array(u_orgim[i]),
                data_format = "channels_last")
            # Save the labels
            training_utils.save_labels_to_xml(
                configs = configs,
                labels = labels[i],
                path = file_path_xml)
    # Save all of the labeled data to pl dataset.
    for l_images, l_labels in l_dataset:
        for i in range(l_images.shape[0]):
            counter+=1
            file_name = configs.training_type + "-" + str(counter).zfill(6)
            file_path_img = os.path.join(
                pl_dataset_dir,
                file_name + ".jpg")
            file_path_xml = os.path.join(
                pl_dataset_dir,
                file_name + ".xml")
            # Save the image
            tf.keras.utils.save_img(
                path = file_path_img,
                x = np.array(l_images[i]),
                data_format = "channels_last")
            # Save the labels
            training_utils.save_labels_to_xml(
                configs = configs,
                labels = l_labels["l"],
                path = file_path_xml)

    # Step 6: Clear the backend and begin the new training for student.
    tf.keras.backend.clear_session()

    # Create the student dataset pipeline.

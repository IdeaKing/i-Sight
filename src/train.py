import os
import shutil
import tensorflow as tf

from src.models import backbones, efficientdet
from src.losses import loss, pseudo_labels
from src.utils import learning_rate

from typing import List, Tuple


class Train:
    def __init__(self,
                 training_dir: str,
                 epochs: int,
                 total_steps: int,
                 input_shape: Tuple[int, int] = (512, 512),
                 precision: str = "float32",
                 training_type: str = "object_detection",
                 max_checkpoints: int = 10,
                 checkpoint_frequency: int = 10,
                 save_model_frequency: int = 10,
                 print_loss: bool = True,
                 log_every_step: int = 100,
                 from_pretrained: str = "",
                 from_checkpoint: bool = False):
        """ Trains the model, either using SSL or SL.

        Params:
            training_dir (str): The filepath to the training directory
            epochs (int): The number of epochs to train the model
            total_steps (int): The total number of steps
            precision (str): Can either be "float32" or "mixed_float16"
            training_type (str): Can be "object_detection", "segmentation", or "classification"
            max_checkpoints (int): The total number of checkpoints to save
        """
        # Initialize the directories
        if os.path.exists(training_dir) and from_checkpoint == False:
            # Prevents accidental deletions
            input("Press Enter to delete the current directory and continue.")
            shutil.rmtree(training_dir)
        else:
            os.makedirs(training_dir)

        # Tensorboard Logging
        tensorboard_dir = os.path.join(
            training_dir, "tensorboard")
        if os.path.exists(tensorboard_dir) is False:
            os.makedirs(tensorboard_dir)
        tensorboard_file_writer = tf.summary.create_file_writer(
            tensorboard_dir)
        tensorboard_file_writer.set_as_default()

        # Define the checkpoint directories
        self.teacher_checkpoint_dir = os.path.join(
            training_dir, "teacher")
        self.tutor_checkpoint_dir = os.path.join(
            training_dir, "tutor")
        self.ema_checkpoint_dir = os.path.join(
            training_dir, "ema")
        self.checkpoint_dir = os.path.join(
            training_dir, "model")
        # Define the full model directories
        self.tutor_exported_dir = os.path.join(
            training_dir, "tutor-exported")
        self.ema_exported_dir = os.path.join(
            training_dir, "ema-exported")
        self.exported_dir = os.path.join(
            training_dir, "model-exported")

        self.epochs = epochs
        self.total_steps = total_steps
        self.steps_per_epoch = int(self.total_steps/self.epochs)
        self.input_shape = input_shape
        self.precision = precision
        self.training_type = training_type
        self.max_checkpoints = max_checkpoints
        self.checkpoint_frequency = checkpoint_frequency
        self.save_model_frequency = save_model_frequency
        self.print_loss = print_loss
        self.log_every_step = log_every_step
        self.from_checkpoint = from_checkpoint
        self.from_pretrained = from_pretrained

        # Semi-supervised configs
        self.mpl_label_smoothing = 0.15
        self.mpl_optimizer_grad_bound = 1e9
        self.uda_label_temperature = 0.7
        self.uda_threshold = 0.5
        self.uda_factor = 8.0  # UDA factor
        self.uda_warmup_steps = 500  # Number of warmup steps for UDA
        self.ema_decay = 0.995
        self.ema_start = 0
        self.tutor_learning_rate = 0.05
        self.tutor_learning_rate_warmup = 50
        self.tutor_learning_rate_numwait = 50
        self.teacher_learning_rate = 0.08
        self.teacher_learning_rate_warmup = 50
        self.teacher_learning_rate_numwait = 0

    def supervised(self,
                   dataset: tf.data.Dataset,
                   model: tf.keras.models.Model,
                   optimizer: tf.keras.optimizers.Optimizer,
                   losses: List or tf.keras.losses.Loss):
        """Supervised training on the model."""
        # Checkpointing Functions
        checkpoint = tf.train.Checkpoint(
            model=model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            self.checkpoint_dir,
            self.max_checkpoints)

        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                preds = model(x, training=True)
                # Run losses
                if isinstance(losses, list):
                    loss = 0
                    for loss_func in losses:
                        loss += loss_func(y_true=y, y_pred=preds)
                else:
                    loss = losses(y_true=y, y_pred=preds)
                if self.precision == "mixed_float16":
                    loss = optimizer.get_scaled_loss(loss)
            gradients = tape.gradient(
                target=loss,
                sources=model.trainable_variables)
            if self.precision == "mixed_float16":
                gradients = optimizer.get_unscaled_gradients(gradients)
            optimizer.apply_gradients(
                grads_and_vars=zip(gradients, model.trainable_variables))
            return loss

        if self.training_type == "object_detection":
            global_step = 0
            for epoch in range(self.epochs):
                print(f"Epoch: {epoch}")
                for step, (images, label_cls, label_bbx) in enumerate(dataset):
                    labels = (label_cls, label_bbx)
                    loss = train_step(images, labels)
                    if self.print_loss:
                        print(f"Epoch {epoch} Step {step}/{self.steps_per_epoch}", \
                              f"loss {loss}")
                    if global_step % self.checkpoint_frequency == 0:
                        checkpoint_manager.save()
                    global_step = global_step + 1

                if epoch % self.save_model_frequency == 0:
                    tf.keras.models.save_model(
                        model, self.exported_dir)

            print("Finished training.")
            return model

    def semisupervised(self,
                       labeled_dataset: tf.data.Dataset,
                       unlabeled_dataset: tf.data.Dataset,
                       model: tf.keras.models.Model,
                       optimizer: tf.keras.optimizers.Optimizer,
                       losses: tf.keras.losses.Loss,
                       learning_rates: dict,
                       batch_size: int,
                       unlabeled_batch_size: int,
                       num_classes: int,
                       from_pretrained: str = "",
                       teacher_warmup_steps: int = 100):
        """Meta Psuedo-Labels SSL Training."""
        # Optimizers
        teacher_optimizer, tutor_optimizer = optimizer
        # Models
        teacher_model, tutor_model, ema_model = model
        # Loss Functions
        loss_func = losses
        uda_loss_func = loss.UDA(
            batch_size=batch_size,
            unlabeled_batch_size=unlabeled_batch_size,
            num_classes=num_classes,
            loss_func=loss_func,
            training_type=self.training_type)
        if self.training_type == "object_detection":
            convert_to_labels = pseudo_labels.PseudoLabelObjectDetection(
                unlabeled_batch_size=unlabeled_batch_size,
                image_dims=self.input_shape)
        # Checkpointing functions
        teacher_checkpoint = tf.train.Checkpoint(
            optimizer=teacher_optimizer,
            model=teacher_model)
        teacher_checkpoint_manager = tf.train.CheckpointManager(
            teacher_checkpoint,
            self.teacher_checkpoint_dir,
            self.max_checkpoints)
        tutor_checkpoint = tf.train.Checkpoint(
            optimizer=tutor_optimizer,
            model=tutor_model)
        tutor_checkpoint_manager = tf.train.CheckpointManager(
            tutor_checkpoint,
            self.tutor_checkpoint_dir,
            self.max_checkpoints)
        ema_checkpoint = tf.train.Checkpoint(
            model=ema_model)
        ema_checkpoint_manager = tf.train.CheckpointManager(
            ema_checkpoint,
            self.ema_checkpoint_dir,
            self.max_checkpoints)
        # Restores models from checkpoints
        if self.from_checkpoint and from_pretrained != "":
            teacher_checkpoint.restore().expect_partial()
            tutor_checkpoint.restore().expect_partial()
            ema_checkpoint.restore().expect_partial()
        elif from_pretrained != "":
            teacher_model.load_weights(from_pretrained)
            tutor_model.load_weights(from_pretrained)

        #@tf.function
        def train_step(images, labels, step):
            preds = {}
            loss_vals = {}

            # Warm up training for the teacher, typically needed for
            # object detection and segmentation.
            if step < teacher_warmup_steps:
                with tf.GradientTape() as teacher_tape:
                    preds["teacher"] = teacher_model(images["l"], training=True)
                    loss_vals["teacher"] = loss_func(y_true=labels["l"],
                                                     y_pred=preds["teacher"])
                teacher_grad = teacher_tape.gradient(
                    loss_vals["teacher"], teacher_model.trainable_variables)
                teacher_grad, _ = tf.clip_by_global_norm(
                    teacher_grad, self.mpl_optimizer_grad_bound)
                if self.precision == "mixed_float16":
                    teacher_grad = teacher_optimizer.get_unscaled_gradients(
                        teacher_grad)
                teacher_optimizer.apply_gradients(
                    zip(teacher_grad, teacher_model.trainable_variables))
                return loss_vals
            # MPL Pipeline
            else:
                # Step 1: Train on teacher
                with tf.GradientTape() as teacher_tape:
                    all_logits = teacher_model(
                        images["all"],
                        training=True)
                    logits, labels, masks, loss_vals = uda_loss_func(
                        y_true=labels,
                        y_pred=all_logits)
                # Change teacher outputs into pseudo-labels
                labels["u_aug"] = convert_to_labels(logits=logits["u_aug"])

                # Step 2 Run on tutor
                with tf.GradientTape() as tutor_tape:
                    logits["tu_on_u_aug_and_l"] = tutor_model(
                        images["u"],
                        training=True)
                    if self.training_type == "object_detection":
                        tu_on_u_cls, tu_on_aug_cls = tf.split(
                            logits["tu_on_u_aug_and_l"][0],
                            [unlabeled_batch_size, batch_size],
                            axis=0)
                        tu_on_u_bbx, tu_on_aug_bbx = tf.split(
                            logits["tu_on_u_aug_and_l"][1],
                            [unlabeled_batch_size, batch_size],
                            axis=0)
                        logits["tu_on_u"] = (tu_on_u_cls, tu_on_u_bbx)
                        logits["tu_on_l_old"] = (tu_on_aug_cls, tu_on_aug_bbx)
                    else:
                        logits["tu_on_u"], logits["tu_on_l_old"] = tf.split(
                            logits["tu_on_u_aug_and_l"],
                            [unlabeled_batch_size, batch_size],
                            axis=0)
                    # Loss between teacher and student
                    loss_vals["tu_on_u"] = loss_func(
                        y_true=labels["u_aug"],
                        y_pred=logits["tu_on_u"])
                    # Loss on labeled data
                    loss_vals["tu_on_l_old"] = loss_func(
                        y_true=labels["l"],
                        y_pred=logits["tu_on_l_old"])
                    if self.precision == "mixed_float16":
                        loss_vals["tu_on_u"] = tutor_optimizer.get_scaled_loss(
                            loss_vals["tu_on_u"])
                tf.print("loss vals", loss_vals)
                tutor_grad_unlabeled = tutor_tape.gradient(
                    loss_vals["tu_on_u"],
                    tutor_model.trainable_variables)
                tutor_grad_unlabeled, _ = tf.clip_by_global_norm(
                    tutor_grad_unlabeled,
                    self.mpl_optimizer_grad_bound)
                if self.precision == "mixed_float16":
                    tutor_grad_unlabeled = tutor_optimizer.get_unscaled_gradients(
                        tutor_grad_unlabeled)
                tutor_optimizer.apply_gradients(
                    zip(tutor_grad_unlabeled, tutor_model.trainable_variables))

                # Step 3 Student on labeled values + dot product calculation
                logits["tu_on_l_new"] = tutor_model(images["l"])
                loss_vals["tu_on_l_new"] = loss_func(
                    y_true=labels["l"],
                    y_pred=logits["tu_on_l_new"])
                dot_product = loss_vals["tu_on_l_new"] - \
                    loss_vals["tu_on_l_old"]
                limit = 3.0**(0.5)
                moving_dot_product = tf.random_uniform_initializer(
                    minval=-limit,
                    maxval=limit)(shape=dot_product.shape)
                moving_dot_product = tf.Variable(
                    initial_value=moving_dot_product,
                    trainable=False,
                    dtype=tf.float32)
                dot_product = dot_product - moving_dot_product
                dot_product = tf.stop_gradient(dot_product)

                # Step 4: Optimize the teacher on teacher and student performance
                with teacher_tape:
                    loss_vals["mpl"] = loss_func(
                        y_true=labels["u_aug"],
                        y_pred=logits["u_aug"])
                    uda_factor = self.uda_factor * tf.math.minimum(
                        1., tf.cast(self.total_steps,
                                    dtype=tf.float32) /
                        float(self.uda_warmup_steps))
                    loss_vals["teacher"] = tf.reduce_sum(
                        loss_vals["u"] * uda_factor +
                        loss_vals["l"] +
                        loss_vals["mpl"] * dot_product)
                    if self.precision == "mixed_float16":
                        teacher_loss = teacher_optimizer.get_scaled_loss(
                            teacher_loss)
                teacher_grad = teacher_tape.gradient(
                    loss_vals["teacher"], teacher_model.trainable_variables)
                teacher_grad, _ = tf.clip_by_global_norm(
                    teacher_grad, self.mpl_optimizer_grad_bound)
                if self.precision == "mixed_float16":
                    teacher_grad = teacher_optimizer.get_unscaled_gradients(
                        teacher_grad)
                teacher_optimizer.apply_gradients(
                    zip(teacher_grad, teacher_model.trainable_variables))
                return loss_vals

        if self.training_type == "object_detection":
            # Object detection training loop
            global_step = 0
            for epoch in range(self.epochs):
                print(f"Epoch {epoch}")
                for step, (lb_images, lb_labels, lb_bboxes) in enumerate(labeled_dataset):
                    or_images, au_images = next(iter(unlabeled_dataset))
                    # Group the data into dicts for easy access during training
                    images = {"all": tf.concat([lb_images, or_images, au_images], axis=0),
                              "u": tf.concat([au_images, lb_images], axis=0),
                              "l": lb_images}
                    labels = {"l": (lb_labels, lb_bboxes)}
                    # Run the training on models
                    loss_vals = train_step(images, labels, global_step)
                    self.update_ema_weights(ema_model=ema_model,
                                            _model=tutor_model,
                                            step=global_step)
                    # Update the learning rates, pseudo labels, tensorboard
                    tutor_optimizer.learning_rate.assign(
                        learning_rate.updated_learning_rate(
                            global_step,
                            learning_rates["tutor_learning_rate"],
                            self.total_steps,
                            learning_rates["tutor_learning_rate_warmup"],
                            learning_rates["tutor_learning_rate_numwait"]))
                    teacher_optimizer.learning_rate.assign(
                        learning_rate.updated_learning_rate(
                            global_step,
                            learning_rates["teacher_learning_rate"],
                            self.total_steps,
                            learning_rates["teacher_learning_rate_warmup"],
                            learning_rates["teacher_learning_rate_numwait"]))
                    self.update_pseudo_labels(pseudo_labeler=convert_to_labels,
                                              step=global_step)
                    if global_step % self.log_every_step == 0:
                        self.update_tensorboard(losses=loss_vals,
                                                step=global_step)
                    if self.print_loss:
                        print(
                            f"Epoch {epoch} Step {int(step+1)}/{self.steps_per_epoch} ", \
                            " ".join(f"loss-{key} {loss}" for key, loss in loss_vals.items()))
                    global_step = global_step + 1

                # Save checkpoints if needed
                if epoch % self.checkpoint_frequency == 0:
                    teacher_checkpoint_manager.save()
                    if global_step > teacher_warmup_steps:
                        tutor_checkpoint_manager.save()
                        ema_checkpoint_manager.save()

                # Saving the model
                if global_step % self.save_model_frequency == 0:
                    tf.keras.models.save_model(
                        tutor_model, self.tutor_exported_dir)
                    if step > teacher_warmup_steps:
                        tf.keras.models.save_model(
                            ema_model, self.ema_exported_dir)
            
            # Save all models after training
            tf.keras.models.save_model(
                teacher_model, self.teacher_exported_dir)
            tf.keras.models.save_model(
                tutor_model, self.tutor_exported_dir)
            tf.keras.models.save_model(
                ema_model, self.ema_exported_dir)
            
            return teacher_model, tutor_model, ema_model

    def update_ema_weights(self, 
                           ema_model: tf.keras.models.Model, 
                           _model: tf.keras.models.Model, 
                           step: int):
        """Update according to ema and return new weights."""
        ema_step = float(step - self.ema_start)
        decay = 1.0 - min(self.ema_decay,
                        (ema_step + 1.0) / (ema_step + 10.0))
        decay = 1.0 if step < self.ema_start else decay
        new_weights = []
        for curr, new in zip(ema_model.get_weights(), _model.get_weights()):
            new_weights.append(curr * (1 - decay) + new * decay)
        ema_model.set_weights(new_weights)
    
    def update_pseudo_labels(self, pseudo_labeler: object, step: int):
        """Updates the bounding box NMS values for the pseudo-labeler.
        As training progresses, the thresholds increase to prevent misclassifications.
        """
        if step == int((0.30*self.total_steps)):
            pseudo_labeler.update_postprocess(
                score_threshold=0.15,
                iou_threshold=0.6)
        elif step == int((0.50*self.total_steps)):
            pseudo_labeler.update_postprocess(
                score_threshold=0.2,
                iou_threshold=0.65)
        elif step == int((0.70*self.total_steps)):
            pseudo_labeler.update_postprocess(
                score_threshold=0.25,
                iou_threshold=0.7)

    def update_tensorboard(self, losses, step):
        # Adds loss information to TB
        for key, value in losses.items():
            tf.summary.scalar(
                key, data=value, step=step)
        tf.summary.flush()

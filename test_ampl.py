import tensorflow as tf

from src.config import Configs
from src.models.efficientdet import model_builder
from src.dataset import Dataset, load_data
from src.losses.loss import effdet_loss, PseudoLabelObjectDetection, UDA
from src.utils.training_utils import learning_rate, update_ema_weights

tf.config.run_functions_eagerly(True)


if __name__ == "__main__":
    # Prepare the configs
    configs = Configs(
        training_type="obd",
        dataset_path="datasets/data/obd_fundus",
        training_dir="test-new-training")
    
    # Prepare the labeled and unlabeled datasets
    u_file_names = load_data(configs, "unlabeled")
    l_file_names = load_data(configs, "labeled")

    # Update the configs
    configs.update_training_configs(
        dataset_size=len(l_file_names),
        unlabeled_data_size=len(u_file_names))

    unlabeled_data = Dataset(
        file_names=u_file_names,
        configs=configs,
        dataset_type="unlabeled").create_dataset()
    labeled_data = Dataset(
        file_names=l_file_names,
        configs=configs,
        dataset_type="labeled").create_dataset()
    
    # Test the dataset outputs
    for x, y in labeled_data:
        print("Labeled image shape: {}".format(x.numpy().shape))
        print("Labeled label shape: {}".format(y.numpy().shape))
        break
    for x, y in unlabeled_data:
        print("Unlabeled image shape: {}".format(x.numpy().shape))
        print("Unlabeled label shape: {}".format(y.numpy().shape))
        break

    # Build the models
    teacher_model = model_builder(
        configs, "teacher")
    tutor_model = model_builder(
        configs, "tutor")
    ema_model = model_builder(
        configs, "ema")
    
    # Optimizers and losses
    loss_func = effdet_loss(configs)
    uda_loss_func = UDA(configs)
    psuedo_label_func = PseudoLabelObjectDetection(configs)

    teacher_optimizer = tf.keras.optimizers.Adam()
    tutor_optimizer = tf.keras.optimizers.Adam()

    # Metrics
    metrics = {"mpl/dot-product": tf.keras.metrics.Mean(),
               "mpl/moving-dot-product": tf.keras.metrics.Mean(),
               "mpl-loss/teacher-on-l": tf.keras.metrics.Mean(),
               "mpl-loss/teacher-on-u": tf.keras.metrics.Mean(),
               "mpl-loss/tutor-on-u": tf.keras.metrics.Mean(),
               "mpl-loss/tutor-on-l": tf.keras.metrics.Mean()}

    # Training function
    def train_step(images, labels, step):
        """Trains one step of AMPL.
        :params images (dict): Contains "all", "unlabeled", and "labeled" images
        :params labels: Array of bounding box annotations
        :returns: A dict of losses.
        """
        
        with tf.GradientTape() as te_tape:
            logits = teacher_model(
                images["l"],
                training=True)
            loss = {"l": loss_func(labels["l"], logits)}

        # Only run AMPL after a certain number of steps
        # This is to prevent the 0-labels problem
        if step > configs.warmup_steps:
        # Step 1: Train on teacher
            with tf.GradientTape() as te_tape:
                all_logits = teacher_model(
                    images["all"],
                    training=True)
                logits, labels, masks, loss = uda_loss_func(
                    y_true=labels,
                    y_pred=tf.cast(all_logits, tf.float32))

            # Change teacher outputs into pseudo-labels
            labels["u_aug"] = psuedo_label_func(
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
                loss["tu_on_u"] = loss_func(
                    y_true=labels["u_aug"],
                    y_pred=tf.cast(logits["tu_on_u"], tf.float32))
                # Loss on labeled data
                loss["tu_on_l_old"] = loss_func(
                    y_true=labels["l"],
                    y_pred=tf.cast(logits["tu_on_l_old"], tf.float32))
                if configs.mixed_precision is True:
                    loss["tu_on_u"] = tutor_optimizer.get_scaled_loss(
                        loss["tu_on_u"])
            try:
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

            return {"mpl/dot-product": dot_product,
                    "mpl/moving-dot-product": moving_dot_product,
                    "mpl-loss/teacher-on-l": loss["teacher"],
                    "mpl-loss/teacher-on-u": loss["u"],
                    "mpl-loss/tutor-on-u": loss["tu_on_u"],
                    "mpl-loss/tutor-on-l": loss["tu_on_l_new"]}
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
            
            return {"mpl/dot-product": 0,
                    "mpl/moving-dot-product": 0,
                    "mpl-loss/teacher-on-l": loss["teacher"],
                    "mpl-loss/teacher-on-u": 0,
                    "mpl-loss/tutor-on-u": 0,
                    "mpl-loss/tutor-on-l": 0}

    global_steps = 0
    for epoch in range(configs.epochs):
        print("Epoch: {}/{}".format(epoch, configs.epochs))
        for step, (lb_image, lb_label) in enumerate(labeled_data):
            # Grab data from unlabeled dataset
            lb_label = lb_label.numpy()
            or_image, au_image = next(iter(unlabeled_data))
            
            # Update the learning rates
            tutor_optimizer.learning_rate.assign(
                learning_rate(
                    global_steps,
                    configs.student_learning_rate,
                    configs.total_steps,
                    configs.tutor_learning_rate_warmup,
                    configs.tutor_learning_rate_numwait))
            teacher_optimizer.learning_rate.assign(
                learning_rate(
                    global_steps,
                    configs.teacher_learning_rate,
                    configs.total_steps,
                    configs.teacher_learning_rate_warmup,
                    configs.teacher_learning_rate_numwait))

            # Group the data into dicts for easy access during training
            images = {"all": tf.concat([lb_image, or_image, au_image], axis=0),
                      "u": tf.concat([au_image, lb_image], axis=0),
                      "l": lb_image}
            labels = {"l": lb_label}
            losses = train_step(images, labels, step)

            update_ema_weights(
                configs,
                ema_model,
                tutor_model,
                step)

            for key, metric in metrics.items():
                metric(losses[key])
            
            print("Epoch: {}/{} Step: {}/{} Teacher-L: {} Tutor-l {}".format(
                epoch, configs.epochs, global_steps, 
                int(configs.total_steps/configs.epochs), 
                metrics["mpl-loss/teacher-on-l"].result(),
                metrics["mpl-loss/tutor-on-l"].result()))
            
            global_steps = global_steps + 1
        
        tf.keras.models.save_model(
            tutor_model, 
            configs.training_dir + "/tutor")
        tf.keras.models.save_model(
            ema_model, 
            configs.training_dir + "/ema")


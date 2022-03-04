from efficien

class EfficientDetNetTrainHub(EfficientDetNetTrain):
  """EfficientDetNetTrain for Hub module."""

  def __init__(self, config, hub_module_url, name=''):
    super(efficientdet_keras.EfficientDetNet, self).__init__(name=name)
    self.config = config
    self.hub_module_url = hub_module_url
    self.base_model = hub.KerasLayer(hub_module_url, trainable=True)

    # class/box output prediction network.
    num_anchors = len(config.aspect_ratios) * config.num_scales

    conv2d_layer = efficientdet_keras.ClassNet.conv2d_layer(
        config.separable_conv, config.data_format)
    self.classes = efficientdet_keras.ClassNet.classes_layer(
        conv2d_layer,
        config.num_classes,
        num_anchors,
        name='class_net/class-predict')

    self.boxes = efficientdet_keras.BoxNet.boxes_layer(
        config.separable_conv,
        num_anchors,
        config.data_format,
        name='box_net/box-predict')

    log_dir = os.path.join(self.config.model_dir, 'train_images')
    self.summary_writer = tf.summary.create_file_writer(log_dir)

  def call(self, inputs, training):
    cls_outputs, box_outputs = self.base_model(inputs, training=training)
    for i in range(self.config.max_level - self.config.min_level + 1):
      cls_outputs[i] = self.classes(cls_outputs[i])
      box_outputs[i] = self.boxes(box_outputs[i])
    return (cls_outputs, box_outputs)
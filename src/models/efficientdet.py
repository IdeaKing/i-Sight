from typing import Any, Union, Tuple, Sequence, Optional

import tensorflow as tf
from src.models.backbone import build_efficient_net_backbone
from src.models.bifpn import BiFPN
from src.models.fpn import FPN
from src.models.head import RetinaNetBBPredictor, RetinaNetClassifier

TrainingOut = Tuple[tf.Tensor, tf.Tensor]
InferenceOut = Tuple[Sequence[tf.Tensor], 
                     Sequence[tf.Tensor], 
                     Sequence[tf.Tensor]]


class EfficientDet(tf.keras.Model):
    """
    Parameters
    ----------
    num_classes: int
        Number of classes to classify
    D: int, default 0
        EfficientDet architecture based on a compound scaling,
        to better understand this parameter refer to EfficientDet 
        paper 4.2 section
    bidirectional: bool, default True
        Use biFPN as feature extractor or FPN. If the value is set to True, then
        a biFPN will be used
    freeze_backbone: bool, default False
        Wether to freeze the efficientnet backbone or not
    score_threshold: float, default 0.1
        Score threshold to give a prediction as valid
    weights: str, default "imagenet"
        If set to "imagenet" then the backbone will be pretrained
        on imagenet. If set to None, the backbone and the bifpn will be random
        initialized. If set to other value, both the backbone and bifpn
        will be initialized with pretrained weights
    training_mode: bool, default False
        If set to True, an extra layer is going to be appended on top of the 
        model. This layer will take care of regress and filter detections.
        Set to True when using model on inference. 
    """
    def __init__(self, 
                 num_classes: Optional[int] = None,
                 configs: object = None, 
                 bidirectional: bool = True,
                 freeze_backbone: bool = False,
                 weights : Optional[str] = "imagenet",
                 training_mode: bool = False) -> None:
                 
        super(EfficientDet, self).__init__()

        # Declare the model architecture
        self.config = configs
        
        # Setup efficientnet backbone
        backbone_weights = "imagenet" if weights == "imagenet" else None
        
        self.backbone = (build_efficient_net_backbone(
            self.config.network, 
            backbone_weights))
        for l in self.backbone.layers:
            l.trainable = not freeze_backbone
        self.backbone.trainable = not freeze_backbone
        
        # self.backbone = models.build_efficient_net_backbone(configs)
        
        # Setup the feature extractor neck
        if bidirectional:
            self.neck = BiFPN(
                self.config.w_bifpn, 
                self.config.d_bifpn, 
                prefix="bifpn/")
        else:
            self.neck = FPN(self.config.w_bifpn)

        # Setup the heads
        if num_classes is None:
            raise ValueError("You have to specify the number of classes.")

        self.num_classes = num_classes
        self.class_head = RetinaNetClassifier(
            self.config.w_bifpn,
            self.config.d_class,
            num_classes=self.num_classes,
            prefix="class_head/")
        self.bb_head = RetinaNetBBPredictor(
            self.config.w_bifpn,
            self.config.d_class,
            prefix="regress_head/")
        
        self.training_mode = training_mode

    @property
    def score_threshold(self) -> float:
        return self.filter_detections.score_threshold
    
    @score_threshold.setter
    def score_threshold(self, value: float) -> None:
        self.filter_detections.score_threshold = value

    def call(self, 
             images: tf.Tensor, 
             training: bool = True) -> Union[TrainingOut, InferenceOut]:
        """
        EfficientDet forward step

        Parameters
        ----------
        images: tf.Tensor
        training: bool
            Wether if model is training or it is in inference mode

        """
        training = training and self.training_mode
        features = self.backbone(images, training=training)
        
        # List of [BATCH, H, W, C]
        bifnp_features = self.neck(features, training=training)

        # List of [BATCH, A, 4]
        bboxes = [self.bb_head(bf, training=training) 
                  for bf in bifnp_features]

        # List of [BATCH, A, num_classes]
        class_scores = [self.class_head(bf, training=training) 
                        for bf in bifnp_features]

        # [BATCH, -1, 4]
        bboxes = tf.concat(bboxes, axis=1)

        # [BATCH, -1, num_classes]
        class_scores = tf.concat(class_scores, axis=1)

        return class_scores, bboxes

    

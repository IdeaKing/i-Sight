import json
from pathlib import Path
from typing import Any, Union, Tuple, Sequence, Optional

import tensorflow as tf
from src import test_models as models


import math
import typing


# D7 the same as D6, therefore we repeat the 6 PHI
PHIs = list(range(0, 7)) + [6]


class AnchorsConfig(typing.NamedTuple):
    sizes: typing.Sequence[int] = (32, 64, 128, 256, 512)
    strides: typing.Sequence[int] = (8, 16, 32, 64, 128)
    ratios: typing.Sequence[float] = (1, 2, .5)
    scales: typing.Sequence[float] = (2 ** 0, 2 ** (1 / 3.0), 2 ** (2 / 3.0))

class EfficientDetBaseConfig(typing.NamedTuple):
    # Input scaling
    input_size: int = 512
    # Backbone scaling
    backbone: int = 0
    # BiFPN scaling
    Wbifpn: int = 64
    Dbifpn: int = 3
    # Box predictor head scaling
    Dclass: int = 3

    def print_table(self, min_D: int = 0, max_D: int = 7) -> None:
        for i in range(min_D, max_D + 1):
            EfficientDetCompudScaling(D=i).print_conf()


class EfficientDetCompudScaling(object):
    def __init__(self, 
                 config : EfficientDetBaseConfig = EfficientDetBaseConfig(), 
                 D : int = 0):
        assert D >= 0 and D <= 7, 'D must be between [0, 7]'
        self.D = D
        self.base_conf = config
    
    @property
    def input_size(self) -> typing.Tuple[int, int]:
        if self.D == 7:
            size = 1536
        else:
            size = self.base_conf.input_size + PHIs[self.D] * 128
        return size, size
    
    @property
    def Wbifpn(self) -> int:
        return int(self.base_conf.Wbifpn * 1.35 ** PHIs[self.D])
    
    @property
    def Dbifpn(self) -> int:
        return self.base_conf.Dbifpn + PHIs[self.D]
    
    @property
    def Dclass(self) -> int:
        return self.base_conf.Dclass + math.floor(PHIs[self.D] / 3)
    
    @property
    def B(self) -> int:
        return self.D
    
    def print_conf(self) -> None:
        print(f'D{self.D} | B{self.B} | {self.input_size:5d} | '
              f'{self.Wbifpn:4d} | {self.Dbifpn} | {self.Dclass} |')

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
    weights: str, default 'imagenet'
        If set to 'imagenet' then the backbone will be pretrained
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
                 D : int = 0, 
                 bidirectional: bool = True,
                 freeze_backbone: bool = False,
                 score_threshold: float = .1,
                 weights : Optional[str] = 'imagenet',
                 custom_head_classifier: bool = False,
                 training_mode: bool = False) -> None:
                 
        super(EfficientDet, self).__init__()

        # Declare the model architecture
        self.config = EfficientDetCompudScaling(D=D)
        
        # Setup efficientnet backbone
        backbone_weights = 'imagenet' if weights == 'imagenet' else None
        self.backbone = (models
                         .build_efficient_net_backbone(
                             self.config.B, backbone_weights))
        for l in self.backbone.layers:
            l.trainable = not freeze_backbone
        self.backbone.trainable = not freeze_backbone
        
        # Setup the feature extractor neck
        if bidirectional:
            self.neck = models.BiFPN(self.config.Wbifpn, self.config.Dbifpn, 
                                     prefix='bifpn/')
        else:
            self.neck = models.FPN(self.config.Wbifpn)

        # Setup the heads
        if num_classes is None:
            raise ValueError('You have to specify the number of classes.')

        self.num_classes = num_classes
        self.class_head = models.RetinaNetClassifier(
            self.config.Wbifpn,
            self.config.Dclass,
            num_classes=self.num_classes,
            prefix='class_head/')
        self.bb_head = models.RetinaNetBBPredictor(self.config.Wbifpn,
                                                   self.config.Dclass,
                                                   prefix='regress_head/')
        
        self.training_mode = training_mode

        # Inference variables, won't be used during training
        self.filter_detections = models.layers.FilterDetections(
            AnchorsConfig(), score_threshold)

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

        # if self.training_mode:
        return class_scores, bboxes
        # else:
        #     return self.filter_detections(images, bboxes, class_scores)
    

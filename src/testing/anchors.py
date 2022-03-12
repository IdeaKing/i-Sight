import tensorflow as tf
import numpy as np

class AnchorGenerator(object):
    
    def __init__(self, 
                 size,
                 aspect_ratios,
                 stride):
        """
        RetinaNet input examples:
            size: 32
            aspect_ratios: [0.5, 1, 2]
        """
        self.size = size
        self.stride = stride

        self.aspect_ratios = aspect_ratios
        self.anchor_scales = [
            2 ** 0,
            2 ** (1 / 3.0),
            2 ** (2 / 3.0)]

        self.anchors = self._generate()
    
    def __call__(self, *args, **kwargs) -> tf.Tensor:
        return self.tile_anchors_over_feature_map(*args, **kwargs)

    def tile_anchors_over_feature_map(
            self, feature_map_shape) -> tf.Tensor:
        """
        Tile anchors over all feature map positions
        Parameters
        ----------
        feature_map: Tuple[int, int, int] H, W , C
            Feature map where anchors are going to be tiled
        
        Returns
        --------
        tf.Tensor of shape [BATCH, N_BOXES, 4]
        """
        def arange(limit: int) -> tf.Tensor:
            return tf.range(0., tf.cast(limit, tf.float32), dtype=tf.float32)
        
        h = feature_map_shape[0]
        w = feature_map_shape[1]

        stride = tf.cast(self.stride, tf.float32)
        shift_x = (arange(w) + 0.5) * stride
        shift_y = (arange(h) + 0.5) * stride

        shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
        shift_x = tf.reshape(shift_x, [-1])
        shift_y = tf.reshape(shift_y, [-1])

        shifts = tf.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
        shifts = tf.transpose(shifts)

        # add A anchors (1, A, 4) to
        # cell K shifts (K, 1, 4) to get
        # shift anchors (K, A, 4)
        # reshape to (K*A, 4) shifted anchors
        A = len(self)
        K = shifts.shape[0]
    
        all_anchors = (tf.reshape(self.anchors, [1, A, 4]) 
                       + tf.cast(tf.reshape(shifts, [K, 1, 4]), tf.float32))
        all_anchors = tf.reshape(all_anchors, [K * A, 4])

        return all_anchors

    def _generate(self) -> tf.Tensor:
        num_anchors = len(self)
        ratios = np.array(self.aspect_ratios)
        scales = np.array(self.anchor_scales)

        # initialize output anchors
        anchors = np.zeros((num_anchors, 4))

        # scale base_size
        anchors[:, 2:] = self.size * np.tile(scales, (2, len(ratios))).T

        # compute areas of anchors
        areas = anchors[:, 2] * anchors[:, 3]

        # correct for ratios
        anchors[:, 2] = np.sqrt(areas / np.repeat(ratios, len(scales)))
        anchors[:, 3] = anchors[:, 2] * np.repeat(ratios, len(scales))

        # transform from (x_ctr, y_ctr, w, h) -> (x1, y1, x2, y2)
        anchors[:, 0::2] -= np.tile(anchors[:, 2] * 0.5, (2, 1)).T
        anchors[:, 1::2] -= np.tile(anchors[:, 3] * 0.5, (2, 1)).T

        return tf.constant(anchors, dtype=tf.float32)

    def __len__(self) -> int:
        return len(self.aspect_ratios) * len(self.anchor_scales)


@tf.function(
    input_signature=[tf.TensorSpec(shape=[None, 4], dtype=tf.float32),
                     tf.TensorSpec(shape=[None, None, None, 3], dtype=tf.float32),
                     tf.TensorSpec(shape=[None, None, 4], dtype=tf.float32),
                     tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                     tf.TensorSpec(shape=None, dtype=tf.int32),
                     tf.TensorSpec(shape=None, dtype=tf.float32),
                     tf.TensorSpec(shape=None, dtype=tf.float32)])
def anchor_targets_bbox(anchors: tf.Tensor,
                        images: tf.Tensor,
                        bndboxes: tf.Tensor,
                        labels: tf.Tensor,
                        num_classes: int,
                        negative_overlap: float = 0.4,
                        positive_overlap: float = 0.5):
    """ 
    Generate anchor targets for bbox detection.
    Parameters
    ----------
    anchors: tf.Tensor 
        Annotations of shape (N, 4) for (x1, y1, x2, y2).
    images: tf.Tensor
        Array of shape [BATCH, H, W, C] containing images.
    bndboxes: tf.Tensor
        Array of shape [BATCH, N, 4] contaning ground truth boxes
    labels: tf.Tensor
        Array of shape [BATCH, N] containing the labels for each box
    num_classes: int
        Number of classes to predict.
    negative_overlap: float, default 0.4
        IoU overlap for negative anchors 
        (all anchors with overlap < negative_overlap are negative).
    positive_overlap: float, default 0.5
        IoU overlap or positive anchors 
        (all anchors with overlap > positive_overlap are positive).
    padding_value: int
        Value used to pad labels
        
    Returns
    --------
    Tuple[tf.Tensor, tf.Tensor]
        labels_batch: 
            batch that contains labels & anchor states 
            (tf.Tensor of shape (batch_size, N, num_classes + 1),
            where N is the number of anchors for an image and the last 
            column defines the anchor state (-1 for ignore, 0 for bg, 1 for fg).
        regression_batch: 
            batch that contains bounding-box regression targets for an 
            image & anchor states (tf.Tensor of shape (batch_size, N, 4 + 1),
            where N is the number of anchors for an image, the first 4 columns 
            define regression targets for (x1, y1, x2, y2) and the
            last column defines anchor states (-1 for ignore, 0 for bg, 1 for fg).
    """
    im_shape = tf.shape(images)
    batch_size = im_shape[0]
    h = tf.cast(im_shape[1], tf.float32)
    w = tf.cast(im_shape[2], tf.float32) 

    result = compute_gt_annotations(anchors, 
                                    bndboxes,
                                    negative_overlap, 
                                    positive_overlap)
    positive_indices, ignore_indices, argmax_overlaps_inds = result

    # Expand ignore indices with out of image anchors
    x_anchor_centre = (anchors[:, 0] + anchors[:, 2]) / 2.
    y_anchor_centre = (anchors[:, 1] + anchors[:, 3]) / 2.

    larger_x = tf.greater_equal(x_anchor_centre, w)
    lesser_x = tf.less(x_anchor_centre, 0)
    out_x = tf.logical_or(larger_x, lesser_x)

    larger_y = tf.greater_equal(y_anchor_centre, h)
    lesser_y = tf.less(y_anchor_centre, 0)
    out_y = tf.logical_or(larger_y, lesser_y)

    out_mask = tf.logical_or(out_x, out_y)
    ignore_indices = tf.logical_or(ignore_indices, out_mask)

    # Gather classification labels
    chose_labels = tf.gather_nd(labels, argmax_overlaps_inds)
    chose_labels = tf.reshape(chose_labels, [batch_size, -1])

    # Labels per anchor 
    # if is positive index add the class, else 0
    # To ignore the label add -1
    labels_per_anchor = tf.where(positive_indices, chose_labels, -1)
    labels_per_anchor = tf.where(ignore_indices, -1, labels_per_anchor)
    labels_per_anchor = tf.one_hot(labels_per_anchor, 
                                   axis=-1, depth=num_classes)
    labels_per_anchor = tf.cast(labels_per_anchor, tf.float32)

    # Add regression for each anchor
    chose_bndboxes = tf.gather_nd(bndboxes, argmax_overlaps_inds)
    chose_bndboxes = tf.reshape(chose_bndboxes, [batch_size, -1, 4])
    regression_per_anchor = bbox_transform(anchors, chose_bndboxes)
    
    # Generate extra label to add the state of the label. 
    # (It should be ignored?)
    indices = tf.cast(positive_indices, tf.float32)
    indices = tf.where(ignore_indices, -1., indices)
    indices = tf.expand_dims(indices, -1)

    labels_per_anchor = tf.concat([labels_per_anchor, indices], axis=-1)
    regression_per_anchor = tf.concat(
        [regression_per_anchor, indices], axis=-1)

    return regression_per_anchor, labels_per_anchor


def compute_gt_annotations(anchors: tf.Tensor,
                           annotations: tf.Tensor,
                           negative_overlap: float = 0.4,
                           positive_overlap: float = 0.5):
    """ 
    Obtain indices of gt annotations with the greatest overlap.
    
    Parameters
    ----------
    anchors: tf.Tensor
        Annotations of shape [N, 4] for (x1, y1, x2, y2).
    annotations: tf.Tensor 
        shape [BATCH, N, 4] for (x1, y1, x2, y2).
    negative_overlap: float, default 0.4
        IoU overlap for negative anchors 
        (all anchors with overlap < negative_overlap are negative).
    positive_overlap: float, default 0.5
        IoU overlap or positive anchors 
        (all anchors with overlap > positive_overlap are positive).
    Returns
    -------
    Tuple[tf.Tensor, tf.Tensor, np.ndarray]
        positive_indices: indices of positive anchors
        ignore_indices: indices of ignored anchors
        argmax_overlaps_inds: ordered overlaps indices
    """
    batch_size = tf.shape(annotations)[0]
    n_anchors = tf.shape(anchors)[0]

    # Cast and reshape inputs to expected values
    anchors = tf.expand_dims(anchors, 0)
    anchors = tf.cast(anchors, tf.float32)
    anchors = tf.tile(anchors, [batch_size, 1, 1])
    annotations = tf.cast(annotations, tf.float32)

    # Compute the ious between boxes, and get the argmax indices and max values
    overlaps = bbox_overlap(anchors, annotations)
    argmax_overlaps_inds = tf.argmax(overlaps, axis=-1, output_type=tf.int32)
    max_overlaps = tf.reduce_max(overlaps, axis=-1)
    
    # Generate index like [batch_idx, max_overlap]	
    batched_indices = tf.ones([batch_size, n_anchors], dtype=tf.int32)
    batched_indices = tf.multiply(tf.expand_dims(tf.range(batch_size), -1),
                                  batched_indices)
    batched_indices = tf.reshape(batched_indices, [-1, 1])
    argmax_inds = tf.reshape(argmax_overlaps_inds, [-1, 1])
    batched_indices = tf.concat([batched_indices, argmax_inds], -1)

    # Assign positive indices. 
    positive_indices = tf.greater_equal(max_overlaps, positive_overlap)
    
    # Assign ignored boxes
    ignore_indices = tf.greater(max_overlaps, negative_overlap)
    ignore_indices = tf.logical_and(ignore_indices, 
                                    tf.logical_not(positive_indices))
    ignore_indices = tf.logical_or(ignore_indices, tf.less(max_overlaps, 0.))

    return positive_indices, ignore_indices, batched_indices


def bbox_transform(anchors: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """Compute bounding-box regression targets for an image."""

    anchors = tf.cast(anchors, tf.float32)
    gt_boxes = tf.cast(gt_boxes, tf.float32)

    Px = (anchors[..., 0] + anchors[..., 2]) / 2.
    Py = (anchors[..., 1] + anchors[..., 3]) / 2.
    Pw = anchors[..., 2] - anchors[..., 0]
    Ph = anchors[..., 3] - anchors[..., 1]

    Gx = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2.
    Gy = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2.
    Gw = gt_boxes[..., 2] - gt_boxes[..., 0]
    Gh = gt_boxes[..., 3] - gt_boxes[..., 1]

    tx = (Gx - Px) / Pw
    ty = (Gy - Py) / Ph
    tw = tf.math.log(Gw / Pw)
    th = tf.math.log(Gh / Ph)
    
    targets = tf.stack([tx, ty, tw, th], axis=-1)
    
    return targets


def bbox_overlap(boxes: tf.Tensor, gt_boxes: tf.Tensor) -> tf.Tensor:
    """
    Calculates the overlap between proposal and ground truth boxes.
    Some `gt_boxes` may have been padded. The returned `iou` tensor for these
    boxes will be -1.
    
    Parameters
    ----------
    boxes: tf.Tensor with a shape of [batch_size, N, 4]. 
        N is the number of proposals before groundtruth assignment. The
        last dimension is the pixel coordinates in [xmin, ymin, xmax, ymax] form.
    gt_boxes: tf.Tensor with a shape of [batch_size, MAX_NUM_INSTANCES, 4]. 
        This tensor might have paddings with a negative value.
    
    Returns
    -------
    tf.FloatTensor 
        A tensor with as a shape of [batch_size, N, MAX_NUM_INSTANCES].
    """
    bb_x_min, bb_y_min, bb_x_max, bb_y_max = tf.split(
        value=boxes, num_or_size_splits=4, axis=2)
    gt_x_min, gt_y_min, gt_x_max, gt_y_max = tf.split(
        value=gt_boxes, num_or_size_splits=4, axis=2)

    # Calculates the intersection area.
    i_xmin = tf.math.maximum(bb_x_min, tf.transpose(gt_x_min, [0, 2, 1]))
    i_xmax = tf.math.minimum(bb_x_max, tf.transpose(gt_x_max, [0, 2, 1]))
    i_ymin = tf.math.maximum(bb_y_min, tf.transpose(gt_y_min, [0, 2, 1]))
    i_ymax = tf.math.minimum(bb_y_max, tf.transpose(gt_y_max, [0, 2, 1]))
    i_area = (tf.math.maximum(i_xmax - i_xmin, 0) * 
              tf.math.maximum(i_ymax - i_ymin, 0))

    # Calculates the union area.
    bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
    gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
    
    # Adds a small epsilon to avoid divide-by-zero.
    u_area = bb_area + tf.transpose(gt_area, [0, 2, 1]) - i_area + 1e-8

    # Calculates IoU.
    iou = i_area / u_area

    # Fills -1 for IoU entries between the padded ground truth boxes.
    gt_invalid_mask = tf.less(
        tf.reduce_max(gt_boxes, axis=-1, keepdims=True), 0.0)
    padding_mask = tf.logical_or(
        tf.zeros_like(bb_x_min, dtype=tf.bool),
        tf.transpose(gt_invalid_mask, [0, 2, 1]))
    iou = tf.where(padding_mask, -tf.ones_like(iou), iou)

    return iou
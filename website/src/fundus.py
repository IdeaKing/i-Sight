"""
i-Sight Fundus Image Analysis.
Created by Thomas Chia v1.0.0 4/4/2022
"""

import math
import copy
import colorsys

import cv2
import sympy as sm
import numpy as np
import tensorflow as tf

from typing import Tuple, List, Union

from .utils.weighted_box_fusion import weighted_boxes_fusion
from .utils.postprocess import FilterDetections


class Fundus:
    """Fundus class for pre and post processing fundus images.
    [1] Applies novel preprocessing techniques to fundus images.
    [2] Locates bounding boxes at local and global levels.
    [3] Filters bounding boxes based on score and IOU at both levels.
    [4] Applies weighted boxes fusion on the ensemble of bounding boxes.
    [5] Locates optic disc and macula.
    [6] Segments the optic disc and optic cup to calculate the cup to disc ratio.
    [7] Locates the Fovea.
    """

    def __init__(self,
                 image: np.array,
                 image_dims: tuple = (512, 512),
                 seg_image_dims: tuple = (120, 120),
                 label_dict: dict = None,
                 score_threshold: float = 0.5,
                 iou_threshold: float = 0.5):
        """Instantiate the fundus class.
        Params:
            image: Path to the Fundus Image
            image_dims: The input dimensions of the prediction model
            seg_image_dims: The input dimensions of the segmentation model
            label_dict: A dictionary containing image and labels
            score_threshold: The score for NMS
            iou_threshold: The IOU for NMS
        """
        self.image = image
        self.image_dims = image_dims
        self.seg_image_dims = seg_image_dims
        self.label_dict = label_dict
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

        self.num_classes = len(self.label_dict)

        self.original_shape = None  # Original image that was only cropped
        self.original_image = None  # Shape of the original image that was cropped
        self.batched_quadrants = None  # List of the processed images
        self.filter_detections = FilterDetections(score_threshold=self.score_threshold,
                                                  iou_threshold=self.iou_threshold,
                                                  image_dims=self.image_dims)

        # Predetermined settings
        self.weighted_box_fusion = True
        if self.weighted_box_fusion == True:
            self.wbf_score_threshold = 0.35

        # Important for post processing
        self.optic_disc_id = int(label_dict["optic_disc"])
        self.macula_id = int(label_dict["macula"])

        # Important for display function
        self.thickness = 1
        self.fontScale = 0.5
        self.fontFace = cv2.FONT_HERSHEY_COMPLEX_SMALL

        # Colors for display function
        hsv_tuples = [(1.0 * x / self.num_classes, 1., 1.)
                      for x in range(self.num_classes)]
        color_pallete = list(
            map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.color_pallete = list(map(lambda x: (
            int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), color_pallete))

        # Images
        self.macula_not_located = "docs/macula_not_located.jpg"
        self.od_not_located = "docs/optic_disc_not_located.jpg"

    def fundus_crop(self, image: np.array) -> np.array:
        """Crops the fundus image out, to remove any noise outside of fundus image.
        Params:
            image: A numpy array of the fundus image
        Returns:
            cropped_masked: A numpy array of the cropped fundus image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 10, 255, cv2.THRESH_BINARY)
        masked_data = cv2.bitwise_and(image, image, mask=thresh)
        contours, _ = cv2.findContours(
            image=thresh,
            mode=cv2.RETR_EXTERNAL,
            method=cv2.CHAIN_APPROX_SIMPLE)
        # Find the largest contour in the image
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w > int(0.65*image.shape[0]) and h > int(0.65*image.shape[1]):
                break
        # Crop the image
        cropped_masked = masked_data[y:y+h, x:x+w]
        return cropped_masked

    def fundus_preprocessing(self, image: np.array) -> np.array:
        """Runs the adapted preprocessing found in:
        An improved luminosity and contrast enhancement
            framework for feature preservation in color fundus images.
        Params:
            image: Cropped input image w/o preprocessing
        Returns:
            image_processed: Image with preprocessing
        """
        image = np.array(image, np.uint8)
        # RGB to HSV
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        H, S, V = cv2.split(image_hsv)
        # Gamma adjustment on V Channel
        gamma_val = 1/2.2
        gamma_tab = [((i / 255) ** gamma_val) * 255 for i in range(256)]
        table = np.array(gamma_tab, np.uint8)
        gamma_V = cv2.LUT(V, table)
        image_hsv_gamma = cv2.merge((H, S, gamma_V))
        # HSV to RGB to LAB
        image_lab = cv2.cvtColor(
            cv2.cvtColor(
                image_hsv_gamma, cv2.COLOR_HSV2RGB),
            cv2.COLOR_RGB2LAB)
        # CLAHE on L Channel
        l, a, b = cv2.split(image_lab)
        clahe = cv2.createCLAHE(
            clipLimit=1,
            tileGridSize=(4, 4))
        l = clahe.apply(l)
        lab_image = cv2.merge((l, a, b))
        # Convert image back to rgb
        image_processed = cv2.cvtColor(lab_image, cv2.COLOR_LAB2RGB)
        return image_processed

    def quadrant_sectioning(self, image: np.array) -> List[np.array]:
        """Breaks the image into main image + 4 quadrants for fundus AI processing.
        Params:
            image: Input fundus image to be quadrant sectioned
        Returns:
            List: List of [5, self.image_dims[0], self.image_dims[1], 3]
        """
        height, width, _ = image.shape
        x_center, y_center = int(width / 2), int(height / 2)
        left, right = image[:, :x_center], image[:, x_center:]
        q1, q2 = right[:y_center, :], left[:y_center, :]
        q4, q3 = right[y_center:, :], left[y_center:, :]
        return [cv2.resize(image, self.image_dims),
                cv2.resize(q1, self.image_dims),
                cv2.resize(q2, self.image_dims),
                cv2.resize(q3, self.image_dims),
                cv2.resize(q4, self.image_dims)]

    def preprocess_image(self) -> Tuple[np.array]:
        """Preprocesses an image. Applies novel cropping technique and CLAHE.
        Returns:
            image: The cropped image with no preprocessing.
            batched_quadrants: [5, self.image_dims[0], self.image_dims[1], 3]
        """
        # Read the image, convert to RGB from BGR
        # image = cv2.imread(self.image)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Run the processing
        image_cropped = self.fundus_crop(self.image)
        image_processed = self.fundus_preprocessing(image_cropped)
        quadrants = self.quadrant_sectioning(image_processed)
        # Save the cropped image dims for post processing
        original_shape = image_cropped.shape
        self.original_shape = (original_shape[1], original_shape[0])
        # TODO: Use Numpy only operations
        # Stack the quadrants
        batched_quadrants = tf.stack(quadrants)
        batched_quadrants = tf.image.resize(images=batched_quadrants,
                                            size=self.image_dims,
                                            method="bilinear")
        batched_quadrants = np.array(batched_quadrants, np.float32)
        # Save the image data into a Fundus attribute
        self.original_image = np.array(image_cropped, np.float32)
        self.batched_quadrants = batched_quadrants

        return batched_quadrants

    def inference_obd(self,
                      model: tf.keras.models.Model,
                      image: np.array):
        """Runs the model on the batch of images."""
        pred_cls, pred_box = model(image, training=False)
        pred_cls = tf.cast(pred_cls, tf.float32)
        pred_box = tf.cast(pred_box, tf.float32)
        return {"bboxes": pred_box, "labels": pred_cls}

    def find_landmarks(self,
                       predictions: tuple,
                       image: np.array) -> dict:
        """Locates landmarks from main image detection. It crops the original image,
        not from the preprocessed image.
        Params:
            predictions: Tuple containing the coordinates, classes and scores
        Returns:
            landmarks: Dict containing the optic_disc and macular information
        """
        image = self.original_image
        landmarks = {"optic_disc": None, "macula": None}
        for i, (bboxes, labels, scores) in enumerate(zip(
                predictions["bboxes"], predictions["labels"], predictions["scores"])):
            if labels == self.optic_disc_id:
                x1, y1, x2, y2 = np.array(bboxes, np.int32)
                # Normalizes the Bbox and Rescales the Bbox to original image
                x1 = int(x1 / self.image_dims[0] * self.original_shape[0]) - int(0.05 * self.original_shape[0])
                x2 = int(x2 / self.image_dims[0] * self.original_shape[0]) + int(0.05 * self.original_shape[0])
                y1 = int(y1 / self.image_dims[1] * self.original_shape[1]) - int(0.05 * self.original_shape[1])
                y2 = int(y2 / self.image_dims[1] * self.original_shape[1]) + int(0.05 * self.original_shape[1])
                # Crop out the optic disc
                cropped_optic_disc = image[y1:y2, x1:x2]
                # We take the optic disc of the highest score
                if landmarks["optic_disc"] is None:
                    landmarks["optic_disc"] = {
                        "image": np.array(cropped_optic_disc, np.uint8),
                        "bbox": (x1, y1, x2, y2),
                        "scores": scores}
                else:
                    if scores > landmarks["optic_disc"]["scores"]:
                        landmarks["optic_disc"] = None
                        landmarks["optic_disc"] = {
                            "image": np.array(cropped_optic_disc, np.uint8),
                            "bbox": (x1, y1, x2, y2),
                            "scores": scores}
                    else:
                        pass
            elif labels == self.macula_id:
                x1, y1, x2, y2 = np.array(bboxes, np.int32)
                # Normalizes the Bbox and Rescales the Bbox to original image
                x1 = int(x1 / self.image_dims[0] * self.original_shape[0]) - int(0.05 * self.original_shape[0])
                x2 = int(x2 / self.image_dims[0] * self.original_shape[0]) + int(0.05 * self.original_shape[0])
                y1 = int(y1 / self.image_dims[1] * self.original_shape[1]) - int(0.05 * self.original_shape[1])
                y2 = int(y2 / self.image_dims[1] * self.original_shape[1]) + int(0.05 * self.original_shape[1])
                # Crop out the macula region
                cropped_macula = image[y1:y2, x1:x2]
                # We take the optic disc of the highest score
                if landmarks["macula"] is None:
                    landmarks["macula"] = {
                        "image": np.array(cropped_macula, np.uint8),
                        "bbox": (x1, y1, x2, y2),
                        "scores": scores}
                else:
                    if scores > landmarks["macula"]["scores"]:
                        landmarks["macula"] = None
                        landmarks["macula"] = {
                            "image": np.array(cropped_macula, np.uint8),
                            "bbox": (x1, y1, x2, y2),
                            "scores": scores}
                    else:
                        pass
            else:
                pass
        return landmarks

    def box_fusion(self,
                   predictions: dict) -> Tuple[np.array, np.array, np.array]:
        """Fuses the detection boxes from the different quadrants.
        Basically, it takes the coordinates found in each quadrant and transforms them
        into the correct relative coordinates. Then it rescales the coordinates back into
        the correct image coordinates. ***However, these scaled coordinates are for the processed
        image shape and not the original cropped image.***
        Params:
            predictions: Dict containing the bboxes, labels, and scores for batched_quadrants
        Returns:
            final_bboxes: Adjusted bboxes
            final_labels: Labels
            final_scores: Scores
        """
        final_bboxes = []
        final_labels = []
        final_scores = []

        for prediction in predictions:
            bboxes, labels, scores = predictions[prediction]["bboxes"], \
                predictions[prediction]["labels"], \
                predictions[prediction]["scores"]
            if prediction == "main_image":
                continue
            if bboxes.shape[0] == 0:
                # If the quadrant has no detection, we just pass the coordinates
                continue
            empty_preds = np.zeros(bboxes.shape)
            # Relative "x" position
            if prediction == "quadrant_1" or prediction == "quadrant_4":
                empty_preds[..., [0, 2]] = (
                    (bboxes[..., [0, 2]] + self.image_dims[0]) / (self.image_dims[0] * 2))
            else:
                empty_preds[..., [0, 2]] = (
                    (bboxes[..., [0, 2]]) / (self.image_dims[0] * 2))
            # Relative "y" position
            if prediction == "quadrant_3" or prediction == "quadrant_4":
                empty_preds[..., [1, 3]] = (
                    (bboxes[..., [1, 3]] + self.image_dims[1]) / (self.image_dims[1] * 2))
            else:
                empty_preds[..., [1, 3]] = (
                    (bboxes[..., [1, 3]]) / (self.image_dims[1] * 2))
            # Readjust the coordinates to scale
            empty_preds[..., [0, 2]] = empty_preds[...,
                                                   [0, 2]] * self.image_dims[0]
            empty_preds[..., [1, 3]] = empty_preds[...,
                                                   [1, 3]] * self.image_dims[1]
            final_bboxes.append(empty_preds)
            final_labels.append(labels)
            final_scores.append(scores)
        # Change list to array and concatenate
        final_bboxes = np.concatenate(final_bboxes, axis=0)
        final_labels = np.concatenate(final_labels, axis=0)
        final_scores = np.concatenate(final_scores, axis=0)
        # Apply weighted box fusion
        if self.weighted_box_fusion:
            boxes_list = [predictions["main_image"]
                          ["bboxes"] / 512, final_bboxes / 512]
            labels_list = [predictions["main_image"]["labels"], final_labels]
            scores_list = [predictions["main_image"]["scores"], final_scores]
            _final_bboxes, _final_scores, _final_labels = weighted_boxes_fusion(
                boxes_list=boxes_list,
                scores_list=scores_list,
                labels_list=labels_list,
                weights=[2, 3],
                iou_thr=self.iou_threshold,
                skip_box_thr=self.wbf_score_threshold)
            _final_bboxes = _final_bboxes * 512

            final_bboxes = []
            final_labels = []
            final_scores = []
            for i, score in enumerate(_final_scores):
                if score < self.wbf_score_threshold:
                    continue
                else:
                    final_bboxes.append(_final_bboxes[i])
                    final_labels.append(_final_labels[i])
                    final_scores.append(_final_scores[i])
            final_bboxes = np.array(final_bboxes)
            final_labels = np.array(final_labels)
            final_scores = np.array(final_scores)

        return final_bboxes, final_labels, final_scores

    def display_predictions(self,
                            bboxes: np.array,
                            labels: np.array,
                            scores: np.array) -> np.array:
        """Displays the bounding boxes on the image."""
        image = self.original_image
        for boxes, label, score in zip(bboxes, labels, scores):
            if score < self.wbf_score_threshold:
                # Ignore padding boxes from tf.image.combined_nms
                continue
            if label == self.optic_disc_id or label == self.macula_id:
                continue
            # The color has to be before label is translated to a string
            c = self.color_pallete[int(label)]
            label = list(self.label_dict.keys())[int(label)]
            text = str(f"{label} {round(float(score), 3)}")
            x1, y1, x2, y2 = np.array(boxes, np.int32)
            # Normalizes the Bbox and Rescales the Bbox to original image
            x1 = int(x1 / self.image_dims[0] * self.original_shape[0]) - 5
            x2 = int(x2 / self.image_dims[0] * self.original_shape[0]) + 5
            y1 = int(y1 / self.image_dims[1] * self.original_shape[1]) - 5
            y2 = int(y2 / self.image_dims[1] * self.original_shape[1]) + 5
            (text_width, text_height), baseline = cv2.getTextSize(text=text,
                                                                  fontFace=self.fontFace,
                                                                  fontScale=self.fontScale,
                                                                  thickness=self.thickness)
            cv2.putText(img=image,
                        text=text,
                        org=(x1 + baseline, y1 + text_height),
                        fontFace=self.fontFace,
                        fontScale=self.fontScale,
                        color=(255, 255, 255))
            cv2.rectangle(img=image,
                          pt1=(x1, y1),
                          pt2=(x2, y2),
                          color=c,
                          thickness=self.thickness)
        return image

    def inference_seg(self,
                      model: tf.keras.models.Model,
                      image: np.array) -> np.array:
        """Segments the Optic Disc and Optic Cup."""
        image = tf.image.resize(image, self.seg_image_dims)
        prediction = model(tf.expand_dims(image, axis=0),
                           training=False)
        prediction = tf.squeeze(prediction, axis=0)
        return prediction

    def postprocess_seg(self,
                        prediction_mask: tf.Tensor) -> tf.Tensor:
        """Post processes the segmentations."""
        output_mask = []
        # Iterate over each class label
        for class_label in range(prediction_mask.shape[-1]):
            class_label = tf.expand_dims(
                prediction_mask[..., class_label], axis=-1)
            class_label = tf.math.round(class_label)
            output_mask.append(class_label)
        output_mask = tf.concat(output_mask, axis=-1)
        return output_mask

    def calculate_max_min_difference(self,
                                     ellipse_points: tuple) -> float:
        """Calculates the difference between the maximum and minimum points.
        Parameters:
            ellipse_points: A tuple of information ((x_center, y_center), (width, height), degrees rotated)
        Returns:
            diff (float): The difference between the maximum y-value and minimum y-value
        """
        x1 = ellipse_points[0][0]  # x-center
        y1 = ellipse_points[0][1]  # y-center
        width = ellipse_points[1][0] / 2
        height = ellipse_points[1][1] / 2
        if width > height:
            ma = width  # Major-axis
            mb = height  # Minor-axis
        else:
            mb = width
            ma = height
        th = ellipse_points[2]  # Angles in degrees (theta)
        th = math.radians(th)  # Convert degrees to radians
        # https://en.wikipedia.org/wiki/Ellipse
        a = ma*ma*math.pow(math.sin(th), 2) + mb*mb*math.pow(math.cos(th), 2)
        b = 2*(mb*mb - ma*ma)*math.sin(th)*math.cos(th)
        c = ma**2*math.pow(math.cos(th), 2) + mb**2*math.pow(math.sin(th), 2)
        d = -2*a*x1 - b*y1
        e = -b*x1 - 2*c*y1
        f = a*x1**2 + b*x1*y1 + c*y1**2 - ma**2*mb**2
        x = sm.symbols("x")
        # The y-value when solved for the derivative
        y = (2*a*x + d) / (-1 * b)
        eq = a*x**2 + b*x*y + c*y**2 + d*x + e*y + f  # The general Quadratic Equation
        # The "zeros" of the equation, represent the maximum and minimum
        zeros = sm.solve(eq, x)
        # Find the half-way point
        hw = 0
        for zero in zeros:
            hw += zero
        hw = hw / len(zeros)
        # Redifine variables and solve for the Y-values
        x = hw
        y = sm.symbols("y")
        eq = a*x**2 + b*x*y + c*y**2 + d*x + e*y + f
        y_val = sm.solve(eq, y)
        # Subtract the max-y and min-y
        max_y = y_val[1]
        min_y = y_val[0]
        diff = max_y - min_y
        return diff

    def cdr(self,
            mask_image: np.array,
            optic_disc_image: np.array) -> Union[float, np.array]:
        """ Calculates the cup to disc ratio following image segmentation.
        See notes on calculations.
        Sources: 
            https://stackoverflow.com/questions/32793703/how-can-i-get-ellipse-coefficient-from-fitellipse-function-of-opencv
            https://math.stackexchange.com/questions/498622/mathematics-max-and-min-x-and-y-values-of-an-ellipse 
            https://en.wikipedia.org/wiki/Ellipse
            https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gaf259efaad93098103d6c27b9e4900ffa 
        Parameters:
            mask_image: The image mask [120, 120, 3]
            optic_disc_image: The optic disc image
        Returns: 
            cdr (float): The cup to disc ratio.
            image (np.array): The array with ellipse approximations and cdr points.
        """
        original_image = copy.deepcopy(optic_disc_image)
        mask_image = cv2.resize(
            np.array(mask_image, np.float32), 
            (original_image.shape[:2][1], 
             original_image.shape[:2][0]))
        mask_image = np.array(mask_image * 255, np.uint8)
        grey_image = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
        od_ret,oc_thresh = cv2.threshold(
            mask_image[..., 0], 76, 76, cv2.THRESH_BINARY)
        oc_ret,od_thresh = cv2.threshold(
            grey_image, 70, 100, cv2.THRESH_BINARY)
        od_contours, od_hierarchy= cv2.findContours(
            od_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        oc_contours, oc_hierarchy = cv2.findContours(
            oc_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in range(len(oc_contours)):
            if oc_contours[contour].shape[0] >= int(0.1*mask_image.shape[0]):
                oc_cnt = oc_contours[contour]
        for contour in range(len(od_contours)):
            if od_contours[contour].shape[0] >= int(0.2*mask_image.shape[0]):
                od_cnt = od_contours[contour] 
        oc_ellipse = cv2.fitEllipse(oc_cnt)
        ellipse = cv2.ellipse(original_image, oc_ellipse, (255, 0, 0), 2)
        od_ellipse = cv2.fitEllipse(od_cnt)
        ellipse = cv2.ellipse(original_image, od_ellipse, (0, 0, 255), 2)
        od_length = self.calculate_max_min_difference(od_ellipse)
        oc_length = self.calculate_max_min_difference(oc_ellipse)
        cdr = oc_length/od_length
        return cdr, original_image

    def draw_fovea(self, image: np.array) -> np.array:
        """Draws the fovea on the image.
        Params:
            image (np.array): The image to draw the fovea on.
        Returns:
            image (np.array): The image with the fovea drawn on it.
        """
        image = copy.deepcopy(image)
        # Coordinates of the fovea
        line1_start = (int(image.shape[0] / 2 + image.shape[0] * 0.05),
                       int(image.shape[1] / 2 + image.shape[1] * 0.05))
        line1_end = (int(image.shape[0] / 2 - image.shape[0] * 0.05),
                     int(image.shape[1] / 2 - image.shape[1] * 0.05))
        line2_start = (int(image.shape[0] / 2 - image.shape[0] * 0.05),
                       int(image.shape[1] / 2 + image.shape[1] * 0.05))
        line2_end = (int(image.shape[0] / 2 + image.shape[0] * 0.05),
                     int(image.shape[1] / 2 - image.shape[1] * 0.05))
        cv2.line(image, line1_start, line1_end, (0, 0, 255), 1)
        cv2.line(image, line2_start, line2_end, (0, 0, 255), 1)
        return image

    def postprocess_obd(self, raw_boxes: tf.Tensor, raw_preds: tf.Tensor):
        """Postprocesses the detections.
        Step 1: Split the images: [main_image, quadrants 0 to 3]
        Step 2: Locate the optic disc and macula in main_image
        Step 3: Merge bounding boxes from each quadrant, then apply weighted box fusion
        Step 4: Apply bounding boxes to the main original image
        Step 5: Return the main image, tuple of landmarks (optic disc, macula), new bounding boxes
        Params:
            raw_boxes: Tensor of shape [5, num_anchors, 4]
            raw_preds: Tensor of shape [5, num_anchors, num_classes]
        Returns:
            image: The original image with labels.
            landmarks: Dict of {optic_disc, macula}
            boxes: Tensor of shape [num_boxes, 4]
            labels: Tensor of shape [num_boxes]
            scores: Tensor of shape [num_boxes]
        """
        # Split the images
        image = self.original_image
        labels, bboxes, scores, valid_detections = self.filter_detections(
            labels=raw_preds, bboxes=raw_boxes)
        # Convert the boxes and preds into numpy arrays.
        preds = {
            "main_image": {"bboxes": np.array(bboxes[0][:valid_detections[0]]),
                           "labels": np.array(labels[0][:valid_detections[0]]),
                           "scores": np.array(scores[0][:valid_detections[0]])},
            "quadrant_1": {"bboxes": np.array(bboxes[1][:valid_detections[1]]),
                           "labels": np.array(labels[1][:valid_detections[1]]),
                           "scores": np.array(scores[1][:valid_detections[1]])},
            "quadrant_2": {"bboxes": np.array(bboxes[2][:valid_detections[2]]),
                           "labels": np.array(labels[2][:valid_detections[2]]),
                           "scores": np.array(scores[2][:valid_detections[2]])},
            "quadrant_3": {"bboxes": np.array(bboxes[3][:valid_detections[3]]),
                           "labels": np.array(labels[3][:valid_detections[3]]),
                           "scores": np.array(scores[3][:valid_detections[3]])},
            "quadrant_4": {"bboxes": np.array(bboxes[4][:valid_detections[4]]),
                           "labels": np.array(labels[4][:valid_detections[4]]),
                           "scores": np.array(scores[4][:valid_detections[4]])}
        }
        # Step 2: Locate the optic disc and macula in main_image
        landmarks = self.find_landmarks(predictions=preds["main_image"],
                                        image=image)
        # Step 3: Merge bounding boxes from each quadrant, then apply weighted box fusion
        if self.weighted_box_fusion:
            bboxes, labels, scores = self.box_fusion(preds)
        else:
            # This is if not using box fusion
            labels = preds["main_image"]["labels"]
            bboxes = preds["main_image"]["bboxes"]
            scores = preds["main_image"]["scores"]
        # Step 5: Return the main image, tuple of landmarks (optic disc, macula), new bounding boxes
        image = self.display_predictions(bboxes, labels, scores)
        return {"image": image,
                "landmarks": landmarks,
                "bboxes": bboxes,
                "labels": labels,
                "scores": scores}

    def process_localizations(self, labels: list) -> dict:
        """Counts the localizations in the image."""
        label_counts = dict.fromkeys(self.label_dict, 0)
        for label in labels:
            label = list(self.label_dict.keys())[int(label)]
            label_counts[label] += 1
        return label_counts

    def __call__(self,
                 detection_model: tf.keras.models.Model,
                 segmentation_model: tf.keras.models.Model) -> dict:
        """Runs i-Sight Fundus Processing Algorithm."""
        # Preprocess the image
        batched_quadrants = self.preprocess_image()

        # Object Detection
        obd_predictions = self.inference_obd(
            model=detection_model, image=batched_quadrants)
        obd_outputs = self.postprocess_obd(raw_boxes=obd_predictions["bboxes"],
                                           raw_preds=obd_predictions["labels"])

        # Optic Disc Segmentation
        if obd_outputs["landmarks"]["optic_disc"] == None:
            cdr, seg_outputs = 0.00, cv2.imread(self.od_not_located)
            od_found = False
        else:
            seg_predictions = self.inference_seg(model=segmentation_model,
                                                 image=obd_outputs["landmarks"]["optic_disc"]["image"])
            seg_outputs = self.postprocess_seg(seg_predictions)
            cdr, seg_outputs = self.cdr(mask_image=seg_outputs,
                                        optic_disc_image=obd_outputs["landmarks"]["optic_disc"]["image"])
            od_found = True

        # Macula Localization
        if obd_outputs["landmarks"]["macula"] == None:
            fovea = cv2.imread(self.macula_not_located)
            fovea_found = False
        else:
            macula = obd_outputs["landmarks"]["macula"]["image"]
            fovea = self.draw_fovea(image=macula)
            fovea_found = True
        
        # Process the localizations
        detections = self.process_localizations(obd_outputs["labels"])

        return {"main_image": obd_outputs["image"],
                "optic_disc": {"image": seg_outputs,
                               "found": od_found,
                               "cdr": cdr},
                "macula": {"image": fovea,
                           "found": fovea_found},
                "obd_outputs": {"bboxes": obd_outputs["bboxes"],
                                "labels": obd_outputs["labels"],
                                "scores": obd_outputs["scores"]},
                "detections": detections}

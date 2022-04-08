import cv2
import numpy as np

isight_logo = cv2.imread("docs/isight_logo.jpg")


def resize_with_pad(image: np.array,
                    new_shape: tuple,
                    padding_color: tuple = (255, 255, 255)) -> np.array:
    """Maintains aspect ratio and resizes with padding."""
    original_shape = (image.shape[1], image.shape[0])
    ratio = float(max(new_shape))/max(original_shape)
    new_size = tuple([int(x*ratio) for x in original_shape])
    image = cv2.resize(image, new_size)
    delta_w = new_shape[0] - new_size[0]
    delta_h = new_shape[1] - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    image = cv2.copyMakeBorder(
        image, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=padding_color)
    return image


def display_image(image: np.array,
                  output_image: np.array,
                  location: tuple,
                  padding_color: tuple) -> np.array:
    """Displays the image in the specified location."""
    output_image[
        location[0]:location[0]+435,
        location[1]:location[1]+435,
        :] = cv2.copyMakeBorder(
            cv2.resize(
                image, (425, 425)),
            5, 5, 5, 5,
            cv2.BORDER_CONSTANT,
            value=padding_color)
    return output_image


def display_titles(output_image: np.array) -> np.array:
    """Displays the Titles in the specified location."""
    (_, text_height), _ = cv2.getTextSize(text="Empty",
                                          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=1,
                                          thickness=2)
    # Display the titles first
    output_image = cv2.putText(
        output_image,
        "Main Fundus Lesion Localization",
        (540, 240 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA)
    output_image = cv2.putText(
        output_image,
        "Optic Disc Segmentation",
        (540, 680 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA)
    output_image = cv2.putText(
        output_image,
        "Macula and Fovea Localization",
        (540, 1120 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA)
    output_image = cv2.putText(
        output_image,
        "OCT Segmentation",
        (540, 1580 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        2,
        cv2.LINE_AA)
    return output_image


def display_text(output_image: np.array,
                 detections: dict,
                 cdr: float,
                 macula: bool = True,
                 oct: bool = True,
                 line_padding: int = 15,
                 text_color: tuple = (0, 0, 0),
                 text_size: float = 0.8,
                 text_thickness: float = 2) -> np.array:
    """Displays the text in the specified location."""
    (_, text_height), _ = cv2.getTextSize(text="Empty",
                                          fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                          fontScale=text_size,
                                          thickness=text_thickness)
    for i, (key, value) in enumerate(detections.items()):
        output_image = cv2.putText(
            output_image,
            str(key) + ": " + str(value),
            (540, 300 + (i * (text_height + line_padding))),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            text_color,
            text_thickness,
            cv2.LINE_AA)
    output_image = cv2.putText(
        output_image,
        f"Cup to disc ratio of: {str(round(cdr, 3))}",
        (540, 740 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_size,
        text_color,
        text_thickness,
        cv2.LINE_AA)
    output_image = cv2.putText(
        output_image,
        "A healthy cup has a cup to disc",
        (540, 800 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_size,
        text_color,
        text_thickness,
        cv2.LINE_AA)
    output_image = cv2.putText(
        output_image,
        "ratio between 0.3 and 0.4.",
        (540, 840 + text_height),
        cv2.FONT_HERSHEY_SIMPLEX,
        text_size,
        text_color,
        text_thickness,
        cv2.LINE_AA)
    if macula == True:
        output_image = cv2.putText(
            output_image,
            "Macula and fovea were located.",
            (540, 1170 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            text_color,
            text_thickness,
            cv2.LINE_AA)
    else:
        output_image = cv2.putText(
            output_image,
            "Macula and fovea were NOT located.",
            (540, 1170 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            text_color,
            text_thickness,
            cv2.LINE_AA)
    if oct == True:
        output_image = cv2.putText(
            output_image,
            "OCT was used for Fundus",
            (540, 1630 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            text_color,
            text_thickness,
            cv2.LINE_AA)
        output_image = cv2.putText(
            output_image,
            "Cross Verification.",
            (540, 1680 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            text_color,
            text_thickness,
            cv2.LINE_AA)
    else:
        output_image = cv2.putText(
            output_image,
            "OCT was NOT used for Fundus",
            (540, 1630 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            text_color,
            text_thickness,
            cv2.LINE_AA)
        output_image = cv2.putText(
            output_image,
            "Cross Verification.",
            (540, 1680 + text_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            text_size,
            text_color,
            text_thickness,
            cv2.LINE_AA)
    return output_image


def create_output_image(fundus_predictions: dict,
                        oct_predictions: dict,
                        resize_method: str = "no_pad",
                        padding_color: tuple = (255, 255, 255),
                        background_color: tuple = (255, 255, 255),
                        text_color: tuple = (0, 0, 0),
                        meta_data=None) -> np.array:
    """Creates an output image with the predictions.
    Params:
        fundus_predictions: dict with the fundus predictions.
        oct_predicitions: dict with the oct predictions.
        resize_method: str with the resize method, either "no_pad" or "pad".
        padding_color: tuple with the padding color in RGB
        background_color: tuple with the background color in RGB
        text_color: tuple with the text color in RGB
    """
    # Create the background image
    template = np.ones((2000, 1080, 3), np.uint8)
    template[..., 0] = template[..., 0] * background_color[2]
    template[..., 1] = template[..., 1] * background_color[1]
    template[..., 2] = template[..., 2] * background_color[0]

    # Insert the i-Sight Logo
    template[0:192, 270:810, :] = cv2.resize(isight_logo, (540, 192))

    # Insert the Image predictions
    template = display_image(
        cv2.cvtColor(fundus_predictions["main_image"], cv2.COLOR_RGB2BGR),
        template,
        (220, 20),
        padding_color)
    template = display_image(
        cv2.cvtColor(fundus_predictions["optic_disc"]["image"], cv2.COLOR_RGB2BGR),
        template,
        (655, 20),
        padding_color)
    template = display_image(
        cv2.cvtColor(fundus_predictions["macula"]["image"], cv2.COLOR_RGB2BGR),
        template,
        (1090, 20),
        padding_color)
    template = display_image(
        cv2.cvtColor(oct_predictions["main_image"], cv2.COLOR_RGB2BGR),
        template,
        (1525, 20),
        padding_color)

    # Insert the Titles
    template = display_titles(template)

    # Insert the Text
    template = display_text(output_image=template,
                            detections=fundus_predictions["detections"],
                            cdr=fundus_predictions["optic_disc"]["cdr"],
                            macula=fundus_predictions["macula"]["found"],
                            oct=oct_predictions["found"],
                            text_color=text_color)

    return template

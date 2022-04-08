import re
import base64

import numpy as np

from io import BytesIO
from PIL import Image


def base64_to_pil(img_base64: bytes) -> np.array:
    """Coverts a base 64 image to a Pillow image.
    Parameters:
        img_base64 (bytes): Bytes base64 image array
    Returns:
        pil_image (np.array): A rray of input byte image
    """
    image_data = re.sub('^data:image/.+;base64,', '', img_base64)
    pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
    pil_image = np.asarray(pil_image)
    return pil_image


def np_to_base64(img_np: np.array) -> bytes:
    """Converts numpy to base64 encoded image.
    Parameters:
        img_np (np.array): image array, numpy
    Returns:
        img_base64 (bytes): Base 64 encoded image
    """
    img = Image.fromarray(img_np.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")

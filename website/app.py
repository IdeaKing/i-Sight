import os
import logging
import datetime

import tensorflow as tf

from flask import (Flask, request, render_template, jsonify)

from src import main
from src.utils import hosting_utils

# Initialize the Tensorflow models
models = {
    "object_detection": tf.keras.models.load_model(
        "models/detection_efficientdet_d4_efficientnet_b4_512_512_precision_mixed_float16_custom"),
    "retina_segmentation": tf.keras.models.load_model(
        "models/segmentation_psp_efficientnet_b4_120_120_precision_float32_refuge"),
    "oct_segmentation": None} # tf.keras.models.load_model("models/oct_segmentation/model")}

# Initialize the Flask Webapp
application = Flask(__name__)

# Check if the log directory exists, if not create it
if not os.path.exists("logs"):
    os.makedirs("logs")

# Logging Data
logging.basicConfig(
    filename=os.path.join(
        "logs", 
        "application-" + str(datetime.datetime.now().strftime("%Y-%m-%d")) + ".log"), 
    level=logging.DEBUG)

# The logging function into a file
@application.after_request
def after_request(response):
    """Logging after every request."""
    logger = logging.getLogger("application.access")
    logger.info(
        "%s [%s] %s %s %s %s %s %s %s",
        request.remote_addr,
        datetime.datetime.utcnow().strftime("%d/%b/%Y:%H:%M:%S.%f")[:-3],
        request.method,
        request.path,
        request.scheme,
        response.status,
        response.content_length,
        request.referrer,
        request.user_agent,
    )
    return response

@application.route("/", methods=["GET"])
def index():
    return render_template("index.html")

# The prediction function
@application.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        fundus_image, oct_image = request.json[0], request.json[1]
        # Decode the inputs from webapp
        fundus_image = hosting_utils.base64_to_pil(fundus_image)
        try:
            oct_image = hosting_utils.base64_to_pil(oct_image)
        except:
            oct_image = None
            print("No OCT image provided")
        # Run the prediction
        prediction = main.predict(fundus_image, oct_image, models)
        # Encode the predictions to json
        prediction = hosting_utils.np_to_base64(prediction)
        return jsonify(result=prediction)

    return None

if __name__ == "__main__":
    application.run(host = "0.0.0.0", port = 1229, debug=True)
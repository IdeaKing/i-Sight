import tensorflow as tf

from flask import (Flask, request, render_template, jsonify)

from src import main, logging
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
# Predict test images using API
import secrets
import keras
import tensorflow as tf
from PIL import Image
import numpy as np
from flasgger import Swagger

from flask import Flask, request
app = Flask(__name__)
app.config['SECRET_KEY'] = secrets.token_hex(15)
swagger = Swagger(app)

# model = keras.models.load_model('./model/model.h5')
def load_model():
	global model
	model = keras.models.load_model('./model/model.h5')
	global graph
	graph = tf.get_default_graph()
load_model()

@app.route('/predict_digit', methods=['POST'])
def predict_digit():
    """Example endpoint returning a prediction of mnist
    ---
    tags:
      - Digit image recognition (MNIST)
    parameters:
      - name: image
        in: formData
        type: file
        required: true
    responses:
      200:
        description: Predicted digit
        schema:
          type: string
    """
    # convert image to array
    im = Image.open(request.files['image'])
    # im2arr = np.array(im).reshape((1, 1, 28, 28)) # for theano backend
    im2arr = np.array(im).reshape((1, 28, 28, 1)) # for tensorflow backend
    with graph.as_default():
        pred_digit = np.argmax(model.predict(im2arr))
    return str(pred_digit)

if __name__ == '__main__':
    app.run()
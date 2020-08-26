import os
import cv2
import math
import flask
import numpy as np

from flask import request, jsonify
from src.unetmodel import UNETModel

model = None
model_shape = (1680, 1024)
app = flask.Flask(__name__)


@app.route('/recognize', methods=['POST'])
def recognize():
    batch_size = model.get_batch_size()
    file_keys = list(request.files.keys())
    batch_count = math.ceil(len(file_keys) / batch_size)

    if batch_count:
        result = []

        for i in range(batch_count):
            start = i * batch_size
            stop = start + batch_size

            images, original_shapes = [], []
            for key in file_keys[start:stop]:
                image = cv2.imdecode(np.frombuffer(request.files[key].read(), np.uint8), cv2.IMREAD_GRAYSCALE)
                original_shapes.append(image.shape)
                images.append(cv2.resize(image, (model_shape[1], model_shape[0])))

            out_masks = model.predict(np.asarray(images))
            for mask, original_shape in zip(out_masks, original_shapes):
                mask = np.squeeze(mask * 255).astype(np.uint8)
                result.append(cv2.resize(mask, dsize=(original_shape[1], original_shape[0])).tolist())

        result = {
            'success': True,
            'result': result
        }
    else:
        result = {
            'success': False
        }

    return jsonify(result)


@app.before_first_request
def before_first_request():
    global model

    weight_location = os.path.join('weights', 'weights.h5')
    model = UNETModel(model_shape, weights=weight_location)


if __name__ == '__main__':
    port = os.environ.get('PORT', 5001)
    host = os.environ.get('HOST', '0.0.0.0')
    app.run(host=host, port=port, use_reloader=False)

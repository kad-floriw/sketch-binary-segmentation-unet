import cv2
import json
import logging
import requests
import numpy as np
import matplotlib.pyplot as plt

hostname, building_port, line_port = '127.0.0.1', '5001', '5002'
line_url = 'http://{hostname}:{line_port}/recognize'.format(hostname=hostname, line_port=line_port)
building_url = 'http://{hostname}:{building_port}/recognize'.format(hostname=hostname, building_port=building_port)


def get_image_mask(img, url):
    _, img_encoded = cv2.imencode('.png', img)

    files = {
        'img.png': ('img.png', img_encoded.tobytes(), 'image/png')
    }

    response = requests.post(url, files=files)

    result = json.loads(response.text)
    mask = np.asarray(result['result'][0]).astype(np.uint8)
    mask_indices = np.where(mask > 127)

    return mask_indices


def test_recognize():
    img_gs = cv2.imread('../train_data/image.tif', cv2.IMREAD_GRAYSCALE)

    img = cv2.cvtColor(img_gs, cv2.COLOR_GRAY2RGB)
    image_masked = img.copy()

    line_mask_indices = get_image_mask(img_gs, line_url)
    image_masked[line_mask_indices[0], line_mask_indices[1]] = [255, 0, 0]

    building_mask_indices = get_image_mask(img_gs, building_url)
    image_masked[building_mask_indices[0], building_mask_indices[1]] = [0, 0, 255]

    plt.imshow(np.hstack((img, image_masked)))
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_recognize()

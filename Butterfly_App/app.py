from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
import logging
import numpy as np
import cv2
from PIL import Image
import sys
import tensorflow as tf
from tensorflow import keras

# adding a logger for info and debug
logger = logging.getLogger('b_app_logger')
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
logger.addHandler(stdout_handler)

# initialize the app
app = Flask(__name__)

# load model and set encodings
butterfly_classes = {32: 'pipevine swallow', 8: 'clodius parnassian', 13: 'eastern coma', 14: 'gold banded',
                     10: 'copper tail',47: 'wood satyr', 38: 'silver spot skipper', 20: 'malachite', 6: 'cabbage white',
                     18: 'julia', 41: 'sootywing', 12: 'crimson patch', 30: 'peacock', 4: 'beckers white',
                     1: 'american snoot', 19: 'large marble', 31: 'pine white', 43: 'straited queen', 26: 'orange tip',
                     21: 'mangrove skipper', 42: 'southern dogface', 33: 'purple hairstreak', 39: 'sixspot burnet',
                     15: 'great eggfly', 2: 'an 88', 44: 'two barred flasher', 5: 'black hairstreak',
                     16: 'grey hairstreak', 36: 'red spotted purple', 37: 'scarce swallow', 45: 'ulyses',
                     3: 'banded peacock', 28: 'painted lady', 35: 'red admiral', 11: 'crecent', 27: 'orchard swallow',
                     17: 'indra swallow', 7: 'chestnut', 48: 'yellow swallow tail', 46: 'viceroy', 49: 'zebra long wing',
                     25: 'orange oakleaf', 23: 'monarch', 34: 'question mark', 0: 'adonis', 29: 'paper kite',
                     24: 'morning cloak', 40: 'skipper', 9: 'clouded sulphur', 22: 'metalmark'}

prediction_model = keras.models.load_model('saved_model.pb')


def prediction_prep(image):
    img_data = Image.open(image)
    img_arr = np.array(img_data)
    resize_img = cv2.resize(img_arr, (128, 128))
    col_corr_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
    return col_corr_img


def convert_data(data):
    new_data = np.array(data, dtype=float)
    data_tf = tf.convert_to_tensor(new_data, np.float32)

    return data_tf


@app.route('/', methods=['POST'])  # accept methods into your url/app, so you can send data to database
def index():
    accepted_img = ['.jpg', '.jpeg', '.png']
    errors = []
    valid_img = False

    if request.method == 'POST':
        new_image = request.form['content']  # get info from input labelled content

        # make sure user uploaded valid image.
        for suffix in accepted_img:
            if new_image.endswith(suffix):
                valid_img = True

        if valid_img:
            try:
                processed_img = prediction_prep(new_image)
                img_array = convert_data(processed_img)
                prediction = prediction_model.predict(np.array([img_array, ]))
                final_pred = np.argmax(prediction, axis=1)
                class_pred = butterfly_classes[final_pred]

                response = {'pred': class_pred}

            except Exception as e:
                logger.error(e)
                return 'There was an issue processing your image.'
        else:
            return 'You have not entered a valid image type. Please try again.'

    # return render_template('index.html')  # renders your html page


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

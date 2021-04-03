from flask import Flask, render_template, url_for, request, redirect
from flask_sqlalchemy import SQLAlchemy
import logging
import numpy as np
import cv2
from PIL import Image
import sys
import tensorflow as tf
from tensorflow import keras

# if app updates don't show hit ctrl +F5 to reset

# adding a logger for info and debug
logger = logging.getLogger('b_app_logger')
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler('.\\logs\\logging.txt', mode='a')
logger.addHandler(stdout_handler)
logger.addHandler(file_handler)

# initialize the app
app = Flask(__name__)

# configure database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///images.db'
db = SQLAlchemy(app)


# for storing images
class Item(db.Model):
    id_ = db.Column(db.Integer, primary_key=True)  # in rework, probably should not use "id" as a var name
    content = db.Column(db.String(500), nullable=False)  # nullable=False means you can't get empty string

    def __repr__(self):
        return '<Item %r>' % self.id


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

prediction_model = keras.models.load_model('model_032421')


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


def make_prediction(image_data):
    processed_img = prediction_prep(image_data)
    img_array = convert_data(processed_img)
    img_prediction = prediction_model.predict(np.array([img_array, ]))
    final_pred = np.argmax(img_prediction, axis=1)
    class_pred = butterfly_classes[final_pred[0]]

    return class_pred


@app.route('/', methods=['GET', 'POST'])  # accept both methods into your url/app, so you can receive data
def index():
    if request.method == 'POST' and request.form:
        accepted_img = ['.jpg', '.jpeg', '.png']
        valid_img = False

        new_image = request.form['content']  # get info from input labelled content
        if '"' in new_image:
            new_image = new_image.replace('"', '')

        # make sure user uploaded valid image.
        for suffix in accepted_img:
            if new_image.endswith(suffix):
                valid_img = True

        if valid_img:
            image_item = Item(content=new_image)
            try:
                db.session.add(image_item)  # add item to db
                db.session.commit()
                return redirect('/return')
            except Exception as e:
                logger.error(e)
                error_message = 'There was an issue processing your image.'
                return render_template('error.html', error_message=error_message)
        else:
            logger.error("Invalid image type entered.")
            error_message = 'You have not entered a valid image type. Please try again.'
            return render_template('error.html', error_message=error_message)

    return render_template('index.html')  # renders your html page


@app.route('/return', methods=['GET', 'POST'])
def return_response():
    if request.method == 'POST':
        return redirect('/')

    if request.method == 'GET':
        try:
            image_files = Item.query.order_by(Item.id_).all()
            image = image_files[-1].content

            class_pred = make_prediction(image)
            response = {'Predicted Species': class_pred}

            return render_template('return.html', response=response)
        except Exception as e:
            logger.error(e)
            error_message = "Unable to process image."
            return render_template('error.html', error_message=error_message)


@app.route('/error', methods=['GET', 'POST'])
def errors():
    if request.method == 'POST':
        return redirect('/')

    return render_template('error.html')


if __name__ == '__main__':
    app.run()

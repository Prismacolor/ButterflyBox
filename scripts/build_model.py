import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.utils import to_categorical
import logging
import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from scripts import dataset_prep

root_path = '../data/Butterflies/Labelled'
img_list = []
label_list = []


def label_data(root_):
    # attache labels to our list of images
    class_names = []
    for root, dirs, files in os.walk(root_):
        print('Labelling data...')
        for file in files:
            if file.endswith('.jpg'):
                img_path = os.path.join(root, file)
                img_path = img_path.replace('\\', '/')

                root_list = img_path.split('/')
                label = root_list[4]
                if label not in class_names:
                    class_names.append(label)

                img_list.append(img_path)
                label_list.append(label)

    return class_names, img_list, label_list


def data_processing(images):
    # process data using the functions we created in our data prep script
    processed_data = []
    error_files = []
    for index, image in enumerate(images):
        try:
            processed_img = dataset_prep.prediction_prep(image)
            processed_data.append(processed_img)
        except Exception as e:
            error_files.append(index)

    return processed_data, error_files


def convert_data(data):
    new_data = np.array(data, dtype=float)
    data_tf = tf.convert_to_tensor(new_data, np.float32)
    return data_tf


def build_model(output):
    # base is a series of convolutional and pooling layers
    # added dropout layers to deactivate certain neurons, helps prevent overfitting
    # final layers are a regular dense network used to flatten and sort data based on probability distributions
    try:
        model = Sequential()
        model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(.25))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Dropout(.25))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2)))

        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        model.add(Dense(50, activation='softmax'))

        return model

    except Exception as e:
        print(e)
        return 0


def training():
    # start by getting class names and labels for each label, check for any errors
    classes, images_list, labels = label_data(root_path)
    image_arrays, errors = data_processing(images_list)
    if len(image_arrays) != len(labels):
        for index, label in enumerate(labels):
            if index in errors:
                labels.remove(labels[index])

    # encode the labels column so it contains numerical values
    encoder = preprocessing.LabelEncoder()
    labels_arr = np.array(labels).reshape(-1, 1)
    labels_arr = encoder.fit_transform(labels_arr)
    output_count = len(classes)

    # set the features/inputs vs the labels, split data into training, validation sets and final testing set
    images_TV_data = {'images': image_arrays[:12000], 'labels': labels_arr[:12000]}
    images_test_data = {'images': image_arrays[12000:], 'labels': labels_arr[12000:]}

    X = images_TV_data['images']
    y = images_TV_data['labels']

    final_test_x = images_test_data['images']
    final_test_y = images_test_data['labels']

    X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, shuffle=True, random_state=42)

    # processing/converting the data sets to the correct formats
    y_train = to_categorical(y, num_classes=50)
    y_test = to_categorical(y, num_classes=50)
    final_test_y = to_categorical(y, num_classes=50)

    X_train, y_train = convert_data(X_train), convert_data(y_train)
    x_test, y_test = convert_data(x_test), convert_data(y_test)
    final_test_x, final_test_y = convert_data(final_test_x), convert_data(final_test_y)

    # now we are going to build the convolutional neural network
    cnn_model = build_model(output_count)

    if cnn_model != 0:
        sess=tf.compat.v1.InteractiveSession()
        print(cnn_model.summary())
        cnn_model.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        cnn_model.fit(X_train, y_train, epochs=10, verbose=2, validation_split=0.2, validation_data=(x_test, y_test))

        # then we'll test our model on our final data set, once it's been tweaked
        sess.close()
        return cnn_model
    else:
        print("Error creating model")
        return 0


# ********* build model script *********
# train model on training set
# evaluate the model on validate set, and tweak
# finally run through test data to get final results
# return the model

# ********* project wrap up *************
# create requirements document
# push to github
# create readme
# package model and deploy either via flask or via docker


if __name__ == '__main__':
    trained_model = training()

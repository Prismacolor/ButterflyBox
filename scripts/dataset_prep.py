import cv2
import numpy as np
from PIL import Image
import os
import logging


# We're going to add to our dataset by flipping our images
# openCV operates with a BGR layout, we'll convert it to RGB so the colors don't change in final image
def data_augmentation(directory, total):
    count = 0

    for root, dirs, files in os.walk(directory):
        print('Processing: ', dirs)
        for file in files:
            if file.endswith('.jpg'):
                count += 1
                print('Flipping', count, 'out of ', total)

                img_path = root.replace('\\', '/') + '/' + file
                img_data = Image.open(img_path)
                img_arr = np.array(img_data)

                new_img = cv2.flip(img_arr, 1)
                new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
                new_img_path = root.replace('\\', '/') + '/flip_' + file
                cv2.imwrite(new_img_path, new_img)

                # this is optional, if you want to add more data.
                # however, performance may lag on regular CPUs given the increased size of your data set
                '''new_img2 = cv2.flip(img_arr, 0)
                new_img2 = cv2.cvtColor(new_img2, cv2.COLOR_BGR2RGB)
                new_img_path2 = root.replace('\\', '/') + '/flip2_' + file
                cv2.imwrite(new_img_path2, new_img2)'''

    data_resize(directory)


# In case we need to remove the extra files we made
def data_removal(directory):
    count2 = 0
    
    for root, dirs, files in os.walk(directory):
        print('Processing: ', root)
        for file in files:
            if 'flip' in file:
                img_path = root.replace('\\', '/') + '/' + file
                os.remove(img_path)


# resize the data for processing
def data_resize(directory):
    count = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                count += 1
                print('Resizing Photo #', count)

                img_path = root.replace('\\', '/') + '/' + file
                img_data = Image.open(img_path)
                img_arr = np.array(img_data)
                resize_img = cv2.resize(img_arr, (128, 128))
                resize_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_path, resize_img)


# this function will prep any prediction images for the model
def prediction_prep(image):
    img_data = Image.open(image)
    img_arr = np.array(img_data)
    resize_img = cv2.resize(img_arr, (128, 128))
    col_corr_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
    return col_corr_img


if __name__ == '__main__':
    img_dir_root = "../Data/Butterflies/Labelled"
    img_total = 4479  # total number of files in root directory, you can remove if you don't need/have this

    data_augmentation(img_dir_root, img_total)
    # data_removal(img_dir_root)

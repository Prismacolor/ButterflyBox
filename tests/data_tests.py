import unittest
from scripts import dataset_prep
import os
import numpy as np
from PIL import Image


class DataAugmentTest(unittest.TestCase):
    def test_data_augment(self):
        self.temp_dir = 'test_files'
        self.total = 3
        self.count = 0

        dataset_prep.data_augmentation(self.temp_dir, self.total)

        for root, dirs, files in os.walk(self.temp_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.count += 1

        self.assertEqual(self.count, 6)


class DataRemovalTest(unittest.TestCase):
    def test_remove_data(self):
        self.temp_dir = 'test_files'
        self.count = 0

        dataset_prep.data_removal(self.temp_dir)

        for root, dirs, files in os.walk(self.temp_dir):
            for file in files:
                if file.endswith('.jpg'):
                    self.count += 1

        self.assertEqual(self.count, 2)


class DataResizeTest(unittest.TestCase):
    def test_resize_data(self):
        self.temp_dir = 'test_files'

        dataset_prep.data_resize(self.temp_dir)

        for root, dirs, files in os.walk(self.temp_dir):
            for file in files:
                if file.endswith('.jpg'):
                    img_path = root.replace('\\', '/') + '/' + file
                    img_data = Image.open(img_path)
                    img_arr = np.array(img_data)
                    self.assertEqual(np.shape(img_arr), (128, 128, 3))

#  MIT License
#
#  Copyright (c) 2020 Omer Ferhat Sarioglu
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.

import os
import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from data import beat_annots, non_beat_annots


class DatasetGenerator:
    def __init__(self, raw_path: str, annot_path: str, beat_width: int, train_ratio=0.8, random_seed=0):
        """
        Custom dataset generator class for arrhythmia classification

        :param raw_path: Raw data path for `X`
        :param annot_path: Annotation data path for `y`
        :param beat_width: Number of samples where beats will be separated
        :param train_ratio: Train test split ratio for validation
        :param random_seed: Randomising condition
        """

        self.split_ratio = train_ratio
        self.beat_width = beat_width
        if random_seed == 0:
            self.random_seed = np.random.randint(0, 1000)
        else:
            self.random_seed = random_seed
        self.raw_path = raw_path
        self.annot_path = annot_path
        # Gets .csv files from folders
        self.raw_files, self.annot_files, self.total_patient = self.get_all_csv()
        # Creates margin in case of failure to meet the number of samples required for the beat
        self.margin = (max(self.beat_width, 64) // 64) + 1
        # Gets data to arrays
        self.raw_data, self.annot_data, self.aux_data = self.get_data_to_arr()
        # Randomise data
        self.shuffle_data()
        # Transpose auxiliary data to get sample to the first column
        self.aux_data = self.aux_data.T
        # Split dataset into test and train
        self.X_train, self.X_test, self.y_train, self.y_test = self.split()

    def get_all_csv(self):
        """
        Take file names from Raw and Annotation folders

        :return: Raw and Annotation file paths and total patient number
        """
        try:
            raw_files = glob.glob(os.path.join(self.raw_path, "*"))
            annot_files = glob.glob(os.path.join(self.annot_path, "*"))
            # Compare number of files
            if len(raw_files) != len(annot_files):
                raise ValueError
            else:
                total_patient = len(raw_files)

            return raw_files, annot_files, total_patient

        except ValueError as e:
            print(e, "Raw files don't match with annotations")

    def get_data_to_arr(self):
        raw_upper_arr = []
        raw_lower_arr = []
        annot_arr = []
        rythm_arr = []
        upper_signal_arr = []
        lower_signal_arr = []

        rythm = None

        for patient in range(self.total_patient):
            raw_data = pd.read_csv(self.raw_files[patient])
            upper_signal_type = raw_data.columns[1].replace("'", "")
            lower_signal_type = raw_data.columns[2].replace("'", "")
            raw_data = raw_data.to_numpy()

            annot_data = pd.read_csv(self.annot_files[patient]).to_numpy()

            for sample, typ, aux in annot_data[:, [2, 3, 7]]:
                if typ in non_beat_annots:
                    if typ == "+":
                        rythm = aux[1:]
                if typ in beat_annots:
                    raw_upper, raw_lower = raw_data[int(sample) - self.beat_width: int(sample) + self.beat_width, 1:3].T
                    if len(raw_upper) == 2 * self.beat_width:
                        raw_upper_arr.append(raw_upper)
                        raw_lower_arr.append(raw_lower)
                        annot_arr.append(typ)
                        rythm_arr.append(rythm)
                        upper_signal_arr.append(upper_signal_type)
                        lower_signal_arr.append(lower_signal_type)

        raw_upper_arr = np.array(raw_upper_arr)
        raw_upper_arr = self.normalize(raw_upper_arr)
        raw_upper_arr = raw_upper_arr[:, :, np.newaxis]

        raw_lower_arr = np.array(raw_lower_arr)
        raw_lower_arr = self.normalize(raw_lower_arr)
        raw_lower_arr = raw_lower_arr[:, :, np.newaxis]

        raw_signal = np.concatenate([raw_upper_arr, raw_lower_arr], axis=-1)

        le_annot = LabelEncoder()
        annot_arr = le_annot.fit_transform(np.array(annot_arr))

        le_rythm = LabelEncoder()
        rythm_arr = le_rythm.fit_transform(np.array(rythm_arr))

        le_signal = LabelEncoder()
        le_signal.fit(np.concatenate([np.array(upper_signal_arr), np.array(lower_signal_arr)]))
        upper_signal_arr = le_signal.transform(np.array(upper_signal_arr))
        lower_signal_arr = le_signal.transform(np.array(lower_signal_arr))

        return raw_signal, annot_arr, np.array([rythm_arr, upper_signal_arr, lower_signal_arr])
        # return raw_signal, annot_arr, rythm_arr

    def shuffle_data(self):
        """
        Randomise all data arrays with random seed if random seed specified.

        :return:
        """
        np.random.seed(self.random_seed)
        np.random.shuffle(self.raw_data)
        np.random.seed(self.random_seed)
        np.random.shuffle(self.annot_data)
        np.random.seed(self.random_seed)
        np.random.shuffle(self.aux_data)

    def split(self):
        X_train = [self.raw_data[:int(len(self.raw_data) * self.split_ratio)],
                   self.aux_data[:int(len(self.aux_data) * self.split_ratio)],]
        Y_train = self.annot_data[:int(len(self.annot_data) * self.split_ratio)]

        X_test = [self.raw_data[int(len(self.raw_data) * self.split_ratio) + 1:],
                  self.aux_data[int(len(self.aux_data) * self.split_ratio) + 1:]]
        Y_test = self.annot_data[int(len(self.annot_data) * self.split_ratio) + 1:]

        return X_train, X_test, Y_train, Y_test

    @staticmethod
    def normalize(arr):
        """
        Standart normalization for array

        :param arr: Numpy array
        :return: Normalized numpy array
        """
        norm_arr = (arr - np.mean(arr)) / np.std(arr)
        return norm_arr

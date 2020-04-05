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

#  MIT License
#
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#
import os
import glob

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from data import beat_annots, non_beat_annots


class DatasetGenerator:
    def __init__(self, raw_path, annot_path, beat_width, random_seed=0):
        self.beat_width = beat_width
        if random_seed == 0:
            self.random_seed = np.random.randint(0, 1000)
        else:
            self.random_seed = random_seed

        self.raw_path = raw_path
        self.annot_path = annot_path
        self.raw_files, self.annot_files, self.total_patient = self.get_all_csv()

        self.margin = (max(self.beat_width, 64) // 64) + 1

        self.raw_data, self.signal_data, self.annot_data, self.rythm_data = self.get_data_to_arr()
        self.shuffle_data()

    def get_all_csv(self):
        try:
            raw_files = glob.glob(os.path.join(self.raw_path, "*"))
            annot_files = glob.glob(os.path.join(self.annot_path, "*"))
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
        upper_signal_type = None
        lower_signal_type = None

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
                    raw_upper, raw_lower = raw_data[int(sample) - self.beat_width: int(sample) + self.beat_width,
                                           1: 3].T
                    if len(raw_upper) == 2 * self.beat_width:
                        raw_upper_arr.append(np.fft.fft(raw_upper))
                        raw_lower_arr.append(np.fft.fft(raw_lower))
                        annot_arr.append(typ)
                        rythm_arr.append(rythm)
                        upper_signal_arr.append(upper_signal_type)
                        lower_signal_arr.append(lower_signal_type)

        raw_upper_arr = np.array(raw_upper_arr)
        raw_upper_arr = raw_upper_arr[:, :, np.newaxis]

        raw_lower_arr = np.array(raw_lower_arr)
        raw_lower_arr = raw_lower_arr[:, :, np.newaxis]

        le_annot = LabelEncoder()
        annot_arr = le_annot.fit_transform(np.array(annot_arr))

        le_rythm = LabelEncoder()
        rythm_arr = le_rythm.fit_transform(np.array(rythm_arr))

        le_upper = LabelEncoder()
        upper_signal_arr = le_upper.fit_transform(np.array(upper_signal_arr))

        le_lower = LabelEncoder()
        lower_signal_arr = le_lower.fit_transform(np.array(lower_signal_arr))

        return [raw_upper_arr, raw_lower_arr], [upper_signal_arr, lower_signal_arr], annot_arr, rythm_arr

    def shuffle_data(self):
        np.random.seed(self.random_seed)
        np.random.shuffle(self.raw_data)
        np.random.seed(self.random_seed)
        np.random.shuffle(self.signal_data)
        np.random.seed(self.random_seed)
        np.random.shuffle(self.annot_data)
        np.random.seed(self.random_seed)
        np.random.shuffle(self.rythm_data)

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

import numpy as np
import tensorflow as tf
from model import model
from data.data_generator import DatasetGenerator

BEAT_WIDTH = 64

model = model.create_model(beat_width=BEAT_WIDTH)

data_generator = DatasetGenerator(raw_path="data/raw",
                                  annot_path="data/annotations/csv",
                                  beat_width=BEAT_WIDTH,
                                  random_seed=100)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="model/logs", histogram_freq=1)

input_signal = data_generator.raw_data
input_aux = np.array(data_generator.rythm_data)

model.fit([input_signal, input_aux], data_generator.annot_data,
          epochs=15, batch_size=50,
          callbacks=[tensorboard_callback])

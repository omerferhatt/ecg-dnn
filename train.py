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
import argparse

from model import model
from data.data_generator import DatasetGenerator


argp = argparse.ArgumentParser()
argp.add_argument("-r", "--raw-path", type=str, default="data/raw",
                  help="Raw signal path.")
argp.add_argument("-a", "--annot-path", type=str, default="data/annotations/csv",
                  help="Path of signal annotations.")
argp.add_argument("-b", "--batch-size", type=int, default=64,
                  help="Number of batch size.")
argp.add_argument("-e", "--epoch", type=int, default=15,
                  help="Number of epoch.")
argp.add_argument("-B", "--beat-width", type=int, default=64,
                  help="Sample number of one beat.")
argp.add_argument("-R", "--random-seed", type=int, default=5,
                  help="Enter `0` for non-random arrays.")
argp.add_argument("-l", "--log-dir", type=str, default="model/logs",
                  help="Folder to save TensorBoard files.")
argp.add_argument("-m", "--model-file", type=str, default="model",
                  help="Name of the *.h5 file.")

args = argp.parse_args()


BEAT_WIDTH = args.beat_width
EPOCH = args.epoch
BATCH_SIZE = args.batch_size

model = model.create_model(beat_width=BEAT_WIDTH)
model.summary()

data_generator = DatasetGenerator(raw_path=args.raw_path,
                                  annot_path=args.annot_path,
                                  beat_width=BEAT_WIDTH,
                                  random_seed=args.random_seed)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=args.log_dir, histogram_freq=1)

model.fit(data_generator.X_train, data_generator.y_train,
          epochs=EPOCH, batch_size=BATCH_SIZE, validation_split=0.1,
          callbacks=[tensorboard_callback])

model.save("model/logs/{}.h5".format(args.model_file))

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

import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam



def create_model(beat_width=64):
    inp_signal = Input(shape=(beat_width * 2, 2), name="input_signal")

    flatten = signal_conv(inp_signal)

    inp_aux = Input(shape=(1,), name="input_aux")

    concat_aux = aux_mlp(flatten, inp_aux)

    ds2 = Dense(512, activation="relu", name="dense_2")(concat_aux)
    drop2 = Dropout(rate=0.18, name="drop_2")(ds2)
    ds3 = Dense(256, activation="relu", name="dense_3")(drop2)
    drop3 = Dropout(rate=0.22, name="drop_3")(ds3)
    ds4 = Dense(128, activation="relu", name="dense_4")(drop3)
    drop4 = Dropout(rate=0.25, name="drop_4")(ds4)
    out_ds5 = Dense(19, activation="softmax", name="output_dense_5")(drop4)

    model = Model(inputs=[inp_signal, inp_aux], outputs=out_ds5, name="ecg_model")
    opt = Adam(0.0001)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def signal_conv(inp):
    c1 = Conv1D(64, kernel_size=9, activation="relu", name="conv1d_1")(inp)
    norm1 = BatchNormalization(name="batch_norm_1")(c1)
    c2 = Conv1D(64, kernel_size=7, activation="relu", name="conv1d_2")(norm1)
    norm2 = BatchNormalization(name="batch_norm_2")(c2)

    mp1 = MaxPooling1D(pool_size=2, strides=2, name="max_pool_1")(norm2)

    c3 = Conv1D(64, kernel_size=7, activation="relu", name="conv1d_3")(mp1)
    norm3 = BatchNormalization(name="batch_norm_3")(c3)
    c4 = Conv1D(128, kernel_size=7, activation="relu", name="conv1d_4")(norm3)
    norm4 = BatchNormalization(name="batch_norm_4")(c4)

    mp2 = MaxPooling1D(pool_size=2, strides=2, name="max_pool_2")(norm4)

    c5 = Conv1D(256, kernel_size=7, activation="relu", name="conv1d_5")(mp2)
    norm5 = BatchNormalization(name="batch_norm_5")(c5)
    c6 = Conv1D(256, kernel_size=7, activation="relu", name="conv1d_6")(norm5)
    norm6 = BatchNormalization(name="batch_norm_6")(c6)

    mp3 = MaxPooling1D(pool_size=2, strides=2, name="max_pool_3")(norm6)

    c7 = Conv1D(512, kernel_size=7, activation="relu", name="conv1d_7")(mp3)
    norm7 = BatchNormalization(name="batch_norm_7")(c7)
    c8 = Conv1D(512, kernel_size=5, activation="relu", name="conv1d_8")(norm7)

    f = Flatten(name="flatten_signal")(c8)
    return f


def aux_mlp(inp, inp_aux):
    con = Concatenate(name="concat_signal_aux")([inp, inp_aux])
    ds1 = Dense(512, activation="relu", name="dense_1")(con)
    drop1 = Dropout(rate=0.16, name="drop_1")(ds1)
    return drop1


class LearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Learning rate scheduler which sets the learning rate according to schedule.

    Arguments:
        schedule: a function that takes an epoch index
            (integer, indexed from 0) and current learning rate
            as inputs and returns a new learning rate as output (float).
    """

    def __init__(self, schedule):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        # Get the current learning rate from model's optimizer.
        lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        tf.keras.backend.set_value(self.model.optimizer.lr, scheduled_lr)
        print('\nEpoch %05d: Learning rate is %6.4f.' % (epoch, scheduled_lr))

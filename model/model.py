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

    ds2 = Dense(256, activation="relu", name="dense_2")(concat_aux)
    drop2 = Dropout(rate=0.18, name="drop_2")(ds2)
    ds3 = Dense(128, activation="relu", name="dense_3")(drop2)
    drop3 = Dropout(rate=0.22, name="drop_3")(ds3)
    ds4 = Dense(64, activation="relu", name="dense_4")(drop3)
    drop4 = Dropout(rate=0.25, name="drop_4")(ds4)
    out_ds5 = Dense(19, activation="softmax", name="output_dense_5")(drop4)

    model = Model(inputs=[inp_signal, inp_aux], outputs=out_ds5, name="ecg_model")
    opt = Adam(0.00005)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def signal_conv(inp):
    c1 = Conv1D(64, kernel_size=7, activation="relu", name="conv1d_1")(inp)
    norm1 = BatchNormalization(name="batch_norm_1")(c1)

    mp1 = MaxPooling1D(pool_size=2, strides=2, name="max_pool_1")(norm1)

    c2 = Conv1D(64, kernel_size=7, activation="relu", name="conv1d_2")(mp1)
    norm2 = BatchNormalization(name="batch_norm_2")(c2)
    c3 = Conv1D(128, kernel_size=7, activation="relu", name="conv1d_3")(norm2)
    norm3 = BatchNormalization(name="batch_norm_3")(c3)

    mp2 = MaxPooling1D(pool_size=2, strides=2, name="max_pool_2")(norm3)

    c4 = Conv1D(256, kernel_size=5, activation="relu", name="conv1d_4")(mp2)

    f = Flatten(name="flatten_signal")(c4)
    return f


def aux_mlp(inp, inp_aux):
    con = Concatenate(name="concat_signal_aux")([inp, inp_aux])
    ds1 = Dense(512, activation="relu", name="dense_1")(con)
    drop1 = Dropout(rate=0.16, name="drop_1")(ds1)
    return drop1

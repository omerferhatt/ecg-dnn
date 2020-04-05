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
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def create_model(beat_width=64):
    inp_upper = Input(shape=(beat_width * 2, 1), name="input_upper")
    inp_lower = Input(shape=(beat_width * 2, 1), name="input_lower")

    flatten_upper = signal_conv(inp_upper)
    flatten_lower = signal_conv(inp_lower)

    inp_upper_aux = Input(shape=(2,))
    inp_lower_aux = Input(shape=(2,))

    ds_upper = aux_mlp(flatten_upper, inp_upper_aux)
    ds_lower = aux_mlp(flatten_lower, inp_lower_aux)

    ds_concat = Concatenate()([ds_upper, ds_lower])
    ds1 = Dense(256, activation="relu")(ds_concat)
    ds2 = Dense(128, activation="relu")(ds1)
    ds3 = Dense(64, activation="relu")(ds2)
    out_ds4 = Dense(19, activation="softmax")(ds3)

    model = Model(inputs=[inp_upper, inp_lower, inp_upper_aux, inp_lower_aux], outputs=out_ds4)
    opt = Adam(0.0005)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def signal_conv(inp):
    c1 = Conv1D(64, kernel_size=7, activation="relu")(inp)
    mp1 = MaxPooling1D(pool_size=2, strides=2)(c1)

    c3 = Conv1D(128, kernel_size=5, activation="relu")(mp1)
    c4 = Conv1D(128, kernel_size=5, activation="relu")(c3)

    mp2 = MaxPooling1D(pool_size=2, strides=2)(c4)
    c5 = Conv1D(256, kernel_size=5, activation="relu")(mp2)

    f = Flatten()(c5)
    return f


def aux_mlp(inp, inp_aux):
    con = Concatenate()([inp, inp_aux])
    ds1 = Dense(256, activation="relu")(con)
    ds2 = Dense(256, activation="relu")(ds1)
    return ds2

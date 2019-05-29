from keras.models import Sequential, Model
from keras.layers import Conv2D, Dense, Input, MaxPooling2D,  UpSampling2D, Reshape, Flatten, Add
import math
kernel_size = (3,3)
strides = (1,1)
padding = "same"
activation = "relu"


def create_encoder_model(width, height, encode_size):
    input_img = Input(shape=(3,width, height))
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation, input_shape = (3, width, height))(input_img)
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = MaxPooling2D(padding=padding)(x)
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = MaxPooling2D(padding=padding)(x)
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = MaxPooling2D(padding=padding)(x)
    x = Flatten()(x)
    encoding = Dense(encode_size)(x)
    return input_img, encoding


def create_decode_model(width, height, encode_size, encoding):
    shape = (int(math.sqrt(encode_size/32)), int(math.sqrt(encode_size/32)), 32)
    x = Reshape(shape)(encoding)
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = Conv2D(32, kernel_size, strides=strides, padding=padding, activation=activation)(x)
    x = UpSampling2D()(x)
    decoding = Conv2D(3, kernel_size, strides=strides, padding=padding, activation="sigmoid")(x)
    return decoding
    

def create_model(width, height):
    encode_size = 512
    input_img, encoding = create_encoder_model(width, height, encode_size)
    modification = Input((encode_size, ))
    added = Add()([encoding, modification])
    decoding = create_decode_model(width, height, encode_size, added)
    model = Model(inputs=[input_img, modification], outputs=[decoding, encoding])
    model.compile(optimizer='rmsprop', loss='mean_squared_error', loss_weights=[1., 0.0])
    return model

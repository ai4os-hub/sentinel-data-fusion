"""
This is a file to test the models we have already trained.

Date: February 2019 & June 2023
Author: Ignacio Heredia & María Peña
Email: iheredia@ifca.unican.es & penam@ifca.unican.es
"""

from __future__ import division
from keras.models import Model
from tensorflow.keras.layers import Input
from keras.layers import (
    Conv2D,
    Concatenate,
    Activation,
    Lambda,
    Add,
    Dropout,
    LeakyReLU,
    Dense,
    Flatten,
    BatchNormalization,
)
from tensorflow.keras.regularizers import l2
import tensorflow as tf


def resBlock(x, channels, kernel_size=[3, 3], scale=0.1, drop_and_batch=True):
    """
    Parameters
    ----------
    x : arr
        Input to the block
    channels : int
        Number of channels of the conv layers
    scale : float
        Scaling factor of the Lambda layer
    """

    tmp = Conv2D(
        channels,
        kernel_size,
        kernel_initializer="he_uniform",
        kernel_regularizer=l2(1e-4),
        padding="same",
    )(x)
    if drop_and_batch:
        tmp = BatchNormalization()(tmp)  # TEMP TEST
    tmp = Activation("relu")(tmp)
    tmp = Conv2D(
        channels,
        kernel_size,
        kernel_initializer="he_uniform",
        kernel_regularizer=l2(1e-4),
        padding="same",
    )(tmp)
    tmp = Lambda(lambda x: x * scale)(tmp)

    return Add()([x, tmp])


def s2model(input_shapes, num_layers=32, feature_size=256, drop_and_batch=True):
    """
    Parameters
    ----------
    input_shapes : dict
    num_layers : int
    feature_size : int

    Returns
    -------
    Keras model

    Notes
    -----
    Original paper by Laharas et al also had a deep version with (num_layers=32, feature_size=256) although we disable
    default as the performance gains were minor in comparison with the shallow one.
    """

    input_list = []
    for (
        res,
        shape,
    ) in input_shapes.items():  # .items sorts by key values (unlike .iteritems)
        input_list.append(Input(shape=shape, name=res))

    x = Concatenate(axis=3)(input_list)  # axis=1
    x = Conv2D(
        feature_size,
        (3, 3),
        kernel_initializer="he_uniform",
        kernel_regularizer=l2(1e-4),
        activation="relu",
        padding="same",
    )(x)
    if drop_and_batch:
        x = Dropout(0.2)(x)  # TEMP TEST

    for i in range(num_layers):
        x = resBlock(x, feature_size, drop_and_batch=drop_and_batch)
        if drop_and_batch:
            x = Dropout(0.1)(x)  # TEMP TEST

    x = Conv2D(
        int(input_list[-1].shape[3]),
        (3, 3),
        kernel_initializer="he_uniform",
        kernel_regularizer=l2(1e-4),
        padding="same",
    )(
        x
    )  # input_list[-1].shape[1]
    x = Add()([x, input_list[-1]])  # add input of the maximum resolution
    model = Model(inputs=input_list, outputs=x)
    return model


models = None
models_sat = None


def discriminator_model(patch_size):
    """
    Discriminator model to use in the GAN
    Parameters
    ----------
    patch_size: tuple
        Shape of the input of the discriminator
    Returns
    -------
    Keras model
    """
    model = tf.keras.Sequential()
    model.add(
        Conv2D(
            64,
            (5, 5),
            strides=(2, 2),
            padding="same",
            input_shape=(patch_size[0], patch_size[1], 21),
        )
    )
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU())
    model.add(Dropout(0.3))

    model.add(Flatten())
    model.add(Dense(1))

    return model

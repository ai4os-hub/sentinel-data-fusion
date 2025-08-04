"""
This is a file to test the models we have already trained.

Date: February 2019 & June 2023
Author: Ignacio Heredia & María Peña
Email: iheredia@ifca.unican.es & penam@ifca.unican.es
"""

from __future__ import division
import numpy as np
from ..techniques.networks import s2model
from skimage.transform import resize
from ..techniques.patches import recompose_images, get_test_patches
import tensorflow as tf


def super_resolve(data_bands, model, downsamples=False, patch_size=128, border=8):
    """

    Parameters
    ----------
    data_bands : dict
    model : Keras model
    patch_size : int
    border : int

    Returns
    -------
    Numpy array with the super-resolved image
    """
    back_norm = data_bands[300]

    # Normalize pixel values  and put image in float32 format
    for res in data_bands.keys():
        data_bands[res] = data_bands[res].astype(np.float32)
        for num_band in range(data_bands[res].shape[-1]):
            data_bands[res][:, :, num_band] = (
                data_bands[res][:, :, num_band]
                - np.mean(data_bands[res][:, :, num_band])
            ) / np.sqrt(np.var(data_bands[res][:, :, num_band]))

    # si estamos reconstruyendo imágenes en escala más baja de la original, podemos pasarle toda la imagen entera
    if downsamples:
        data_up = np.zeros(
            (
                data_bands[10].shape[0],
                data_bands[10].shape[1],
                data_bands[300].shape[-1],
            )
        ).astype(np.float32)
        for num_band in range(data_bands[300].shape[-1]):
            data_up[:, :, num_band] = resize(
                image=data_bands[300][:, :, num_band],
                output_shape=data_bands[10].shape[:2],
                mode="reflect",
            )
        inp = {
            "10": np.expand_dims(data_bands[10], axis=0),
            "300": np.expand_dims(data_up, axis=0),
        }
        image = model.predict(inp)
        images = image[0, :, :, :]

    else:  # por problemas de memoria, si estamos trabajando en escala original, tenemos que ir prediciendo en pequeños conjuntos de patches
        # Get the patches and predict
        patches = get_test_patches(
            data_bands=data_bands, patch_size=patch_size, border=border
        )
        patches = {
            str(res): np.moveaxis(data, source=1, destination=3)
            for res, data in patches.items()
        }
        total_len = len(patches[str(res)])
        chunk_size = 200
        if chunk_size < total_len:
            i_ant = 0
            i = chunk_size
            print("Predicting from %i to %i" % (i_ant, i))
            patches_chunk = {str(res): patches[str(res)][i_ant:i] for res in patches}
            prediction = model.predict(patches_chunk)
            i_ant = i
            for i in range(2 * chunk_size, len(patches[str(res)]), chunk_size):
                print("Predicting from %i to %i" % (i_ant, i))
                patches_chunk = {
                    str(res): patches[str(res)][i_ant:i] for res in patches
                }
                prediction_temp = model.predict(patches_chunk)
                prediction = np.concatenate((prediction, prediction_temp), axis=0)
                i_ant = i
            if i_ant < len(patches[str(res)]):
                print("Predicting from %i to %i" % (i_ant, len(patches[str(res)])))
                patches_chunk = {
                    str(res): patches[str(res)][i_ant : len(patches[str(res)])]
                    for res in patches
                }
                prediction_temp = model.predict(patches_chunk)
                prediction = np.concatenate((prediction, prediction_temp), axis=0)

        # Recompose the image from the patches
        min_res = min(data_bands.keys())
        images = recompose_images(
            prediction, border=border, size=data_bands[min_res].shape
        )

    # Undo the pixel normalization and clip to allowed pixel values
    for num_band in range(images.shape[-1]):
        images[:, :, num_band] = images[:, :, num_band] * np.sqrt(
            np.var(back_norm[:, :, num_band])
        ) + np.mean(back_norm[:, :, num_band])

    return images


def load_model(
    input_shape, model_path, num_layers=32, feature_size=256, drop_and_batch=True
):
    """
    Load Keras model from weights
    ----------
    Parameters
    input_shape : dict
        Shape of the model's inputs
    model_path: str
        Path of the model to be trained/tested
    """
    model = s2model(
        input_shape,
        num_layers=num_layers,
        feature_size=feature_size,
        drop_and_batch=drop_and_batch,
    )
    model.load_weights(model_path)
    return model


cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
mae = tf.keras.losses.MeanAbsoluteError()


def discriminator_loss(real_output, fake_output):
    """
    Computes discriminator error
    -----------
    Parameters
    real_output: arr
        True value
    fake_output: arr
        Generator predicted values
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(real_output, fake_output):
    """
    Computes generator error
    ----------
    Parameters
    real_output: arr
        True values
    fake_output: arr
        Generator predicted values
    """
    return mae(real_output, fake_output)

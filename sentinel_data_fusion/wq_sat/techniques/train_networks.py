"""
This is a file to train the models (GAN and CNN).

Date: February 2019 & June 2023
Author: Ignacio Heredia & María Peña
Email: iheredia@ifca.unican.es & penam@ifca.unican.es
"""

from __future__ import division
import time
import os
import tensorflow as tf
from keras.optimizers import Nadam
from keras import callbacks
from .. import config
from ..techniques.networks import s2model
from ..utils import data_utils
from ..utils.data_utils import data_sequence, NPYDataGenerator
from ..techniques.networks import discriminator_model
from ..techniques.patches import create_patches
from ..utils.model_utils import discriminator_loss, generator_loss
from tqdm import tqdm

# some considerations to the training
callback_list = []
callback_list.append(
    callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=1,
        min_delta=1e-6,
        cooldown=20,
        min_lr=1e-5,
    )
)


# ADDED Nicely formatted time string
def hms_string(sec_elapsed):
    """
    sec_elapsed: int
        Number of seconds
    """
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)


def train_cnn(
    name_model,
    path_model,
    name_patches,
    path_patches,
    tiles_dir=None,
    patches_creation=False,
    num_layers=32,
    feature_size=256,
    epochs=30,
    drop_and_batch=True,
):
    """
    name_model: str
    name_patches: str
    patches_creation: bool
        Whether to create patches or not
    num_layers: int
        Number of residual blocks
    feature_size: int
        Number of neurons on each convolutional layer
    epochs: int
        Number of epochs to train the model
    """

    max_res = 300
    batch_size = 2  # You can configure this. Instead of batch_size when calling a function, just keep the batch_size param
    input_shape = {"10": (None, None, 4), "300": (None, None, 21)}
    model = s2model(
        input_shape,
        num_layers=num_layers,
        feature_size=feature_size,
        drop_and_batch=drop_and_batch,
    )
    nadam = Nadam(
        learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8
    )  # clipvalue=5e-6
    model.compile(
        optimizer=nadam, loss="mean_absolute_error", metrics=["mean_squared_error"]
    )
    model.count_params()

    # cargamos la lista de ficheros de train
    train_tiles = data_utils.load_data_splits(
        splits_dir=os.getcwd(), split_name="train"
    )
    patch_save_dir = os.path.join(path_patches, name_patches)
    if not os.path.isdir(patch_save_dir):
        os.mkdir(patch_save_dir)

    # creamos los patches para el train si lo especificamos
    if patches_creation:
        create_patches(
            train_tiles,
            300,
            data_dir=tiles_dir,
            save_dir=patch_save_dir,
            roi_x_y=None,
            num_patches=None,
        )

    # cargamos el generador para los datos de train
    train_gen = data_sequence(
        tiles=train_tiles,
        patches_dir=patch_save_dir,  # (config.data_path(), name_patches),
        batch_size=batch_size,
        max_res=max_res,
    )

    steps_per_epoch = train_gen.__len__()

    def generator_wrapper(
        train_tiles, max_res, batch_size=32, patches_dir=None, shuffle=True
    ):
        generator = NPYDataGenerator(
            train_tiles, max_res, batch_size, patches_dir, shuffle=False
        )
        for X, y in generator:
            yield X, y

    # Suponiendo que las formas y los tipos de datos de tus características y etiquetas son conocidos y coherentes
    # Ajusta la output_signature para reflejar correctamente estas formas y tipos
    train_gen = tf.data.Dataset.from_generator(
        generator=lambda: generator_wrapper(
            train_tiles, max_res, batch_size, patches_dir=patch_save_dir
        ),  # (config.data_path(), name_patches)),
        output_signature=(
            {
                "10": tf.TensorSpec(shape=(batch_size, 120, 120, 4), dtype=tf.float64),
                "300": tf.TensorSpec(
                    shape=(batch_size, 120, 120, 21), dtype=tf.float64
                ),
            },  # Ajusta el dtype según tus características
            tf.TensorSpec(
                shape=(batch_size, 120, 120, 21), dtype=tf.float64
            ),  # Asegúrate de que el dtype de y sea correcto
        ),
    ).prefetch(tf.data.experimental.AUTOTUNE)

    # cargamos la lista de ficheros de validación
    val_tiles = data_utils.load_data_splits(splits_dir=os.getcwd(), split_name="val")

    # creamos los patches para la validación si lo especificamos
    if patches_creation:
        create_patches(
            val_tiles,
            300,
            data_dir=tiles_dir,
            save_dir=patch_save_dir,
            roi_x_y=None,
            num_patches=None,
        )

    # cargamos el generador para los datos de validación
    val_gen = data_sequence(
        tiles=val_tiles,
        batch_size=batch_size,
        patches_dir=patch_save_dir,  # (config.data_path(), name_patches),
        max_res=max_res,
    )

    val_steps = val_gen.__len__()

    val_gen = tf.data.Dataset.from_generator(
        generator=lambda: generator_wrapper(
            val_tiles, max_res, batch_size, patches_dir=patch_save_dir
        ),  # (config.data_path(), name_patches)),
        output_signature=(
            {
                "10": tf.TensorSpec(shape=(batch_size, 120, 120, 4), dtype=tf.float64),
                "300": tf.TensorSpec(
                    shape=(batch_size, 120, 120, 21), dtype=tf.float64
                ),
            },  # Ajusta el dtype según tus características
            tf.TensorSpec(
                shape=(batch_size, 120, 120, 21), dtype=tf.float64
            ),  # Asegúrate de que el dtype de y sea correcto
        ),
    ).prefetch(tf.data.experimental.AUTOTUNE)

    train_gen = train_gen.repeat()  # Repite el dataset indefinidamente
    val_gen = val_gen.repeat()

    # Launch the training
    t0 = time.time()

    # entrenamos el modelo
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        initial_epoch=0,
        validation_data=val_gen,
        validation_steps=val_steps,
        verbose=1,
        callbacks=callback_list,
    )

    elapsed = time.time() - t0
    print(f"Training time: {hms_string(elapsed)}")
    print("Saving the model to h5...")

    # guardamos el modelo
    fpath = os.path.join(path_model, name_model)
    model.save(fpath)

    print("Finished")

    return history.history


def train_step(
    X, y, generator, discriminator, generator_optimizer, discriminator_optimizer
):
    """
    images: data generator
        Train data generator
    generator: model
    discriminator: model
    generator_optimizer: Nadam object
    discriminator_optimizer: Nadam object
    """
    # print("In train_step")
    # print(X['10'].shape)
    # print(X['300'].shape)
    # print(y.shape)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        # obtenemos la imagen SR con el generador y obtenemos el output del discriminador para esa imagen y la target
        generated_images = generator(X, training=True)
        # print(generated_images.shape)
        real_output = discriminator(y, training=True)
        # print(real_output.shape)
        fake_output = discriminator(generated_images, training=True)
        # print(fake_output.shape)

        # calculamso las pérdidas de generador y discriminador
        gen_loss = generator_loss(y, generated_images)
        # print(gen_loss)
        disc_loss = discriminator_loss(real_output, fake_output)
        # print(disc_loss)
        # calculamos los gradientes del generador y discriminador
        # print(generator.trainable_variables)
        gradients_of_generator = gen_tape.gradient(
            gen_loss, generator.trainable_variables
        )
        # print(gradients_of_generator)
        # print(discriminator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(
            disc_loss, discriminator.trainable_variables
        )
        # print(gradients_of_discriminator)
        # actualizamos el generador y el discriminador
        generator_optimizer.apply_gradients(
            zip(gradients_of_generator, generator.trainable_variables)
        )
        discriminator_optimizer.apply_gradients(
            zip(gradients_of_discriminator, discriminator.trainable_variables)
        )


def train_gan(
    gen_model,
    discr_model,
    path_model,
    name_patches,
    path_patches,
    tiles_dir=None,
    patches_creation=False,
    num_layers=32,
    feature_size=256,
    epochs=30,
    drop_and_batch=True,
):
    """
    gen_model: str
    discr_model: str
    name_patches: str
    patches_creation: bool
        Whether to create patches or not
    num_layers: int
        Number of residual blocks
    feature_size: int
        Number of neurons on each convolutional layer
    epochs: int
        Number of epochs to train the model
    """

    # obtenemos la lista de imágenes de entrenamiento
    train_tiles = data_utils.load_data_splits(
        splits_dir=os.getcwd(), split_name="train"
    )
    save_dir = os.path.join(config.data_path(), name_patches)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    batch_size = 1  # 2
    max_res = 300
    patch_save_dir = os.path.join(path_patches, name_patches)
    # creamos los patches de train si así lo pedimos
    if patches_creation:
        create_patches(
            train_tiles,
            300,
            data_dir=tiles_dir,
            save_dir=patch_save_dir,
            roi_x_y=None,
            num_patches=None,
        )

    # obtenemos el generador de datos
    dataset = data_sequence(
        tiles=train_tiles,
        patches_dir=patch_save_dir,
        batch_size=batch_size,
        max_res=max_res,
    )

    dataset.__len__()

    def generator_wrapper(
        train_tiles, max_res, batch_size=32, patches_dir=None, shuffle=True
    ):
        generator = NPYDataGenerator(
            train_tiles, max_res, batch_size, patches_dir, shuffle=False
        )
        for X, y in generator:
            yield X, y

    # Suponiendo que las formas y los tipos de datos de tus características y etiquetas son conocidos y coherentes
    # Ajusta la output_signature para reflejar correctamente estas formas y tipos
    dataset = tf.data.Dataset.from_generator(
        generator=lambda: generator_wrapper(
            train_tiles, max_res, batch_size, patches_dir=patch_save_dir
        ),
        output_signature=(
            {
                "10": tf.TensorSpec(shape=(batch_size, 120, 120, 4), dtype=tf.float64),
                "300": tf.TensorSpec(
                    shape=(batch_size, 120, 120, 21), dtype=tf.float64
                ),
            },  # Ajusta el dtype según tus características
            tf.TensorSpec(
                shape=(batch_size, 120, 120, 21), dtype=tf.float64
            ),  # Asegúrate de que el dtype de y sea correcto
        ),
    ).prefetch(tf.data.experimental.AUTOTUNE)

    # cargamos la lista de ficheros de validación
    val_tiles = data_utils.load_data_splits(splits_dir=os.getcwd(), split_name="val")

    # creamos los patches para la validación si lo especificamos
    if patches_creation:
        create_patches(
            val_tiles,
            300,
            data_dir=tiles_dir,
            save_dir=patch_save_dir,
            roi_x_y=None,
            num_patches=None,
        )

    # cargamos el generador para los datos de validación
    val_gen = data_sequence(
        tiles=val_tiles,
        batch_size=batch_size,
        patches_dir=patch_save_dir,
        max_res=max_res,
    )

    val_gen.__len__()

    val_gen = tf.data.Dataset.from_generator(
        generator=lambda: generator_wrapper(
            val_tiles, max_res, batch_size, patches_dir=patch_save_dir
        ),
        output_signature=(
            {
                "10": tf.TensorSpec(shape=(batch_size, 120, 120, 4), dtype=tf.float64),
                "300": tf.TensorSpec(
                    shape=(batch_size, 120, 120, 21), dtype=tf.float64
                ),
            },  # Ajusta el dtype según tus características
            tf.TensorSpec(
                shape=(batch_size, 120, 120, 21), dtype=tf.float64
            ),  # Asegúrate de que el dtype de y sea correcto
        ),
    ).prefetch(tf.data.experimental.AUTOTUNE)

    dataset = dataset.repeat()  # Repite el dataset indefinidamente
    val_gen = val_gen.repeat()

    # inicializamos la gan
    generator = s2model(
        {"10": (None, None, 4), "300": (None, None, 21)},
        num_layers=num_layers,
        feature_size=feature_size,
        drop_and_batch=drop_and_batch,
    )
    discriminator = discriminator_model((120, 120))
    # nadam = Nadam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
    generator_optimizer = Nadam(
        learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8
    )
    discriminator_optimizer = Nadam(
        learning_rate=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8
    )
    generator.compile(
        optimizer=generator_optimizer,
        loss="mean_absolute_error",
        metrics=["mean_squared_error"],
    )
    discriminator.compile(
        optimizer=discriminator_optimizer,
        loss="binary_crossentropy",
        metrics=["mean_squared_error"],
    )

    # para cada época llamamos a la función anterior para predecir y actualizar pesos
    for epoch in tqdm(range(epochs)):
        start = time.time()
        for X, y in dataset:
            train_step(
                X,
                y,
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer,
            )

        if epoch % 5 == 0:
            if not os.path.isdir(os.path.join(path_model, "backup_gan_models")):
                os.mkdir(os.path.join(path_model, "backup_gan_models"))

            fpath_g = os.path.join(
                path_model, "backup_gan_models", "generator_epoch{}.h5".format(epoch)
            )
            generator.save(fpath_g)

            fpath_d = os.path.join(
                path_model,
                "backup_gan_models",
                "discriminator_epoch{}.h5".format(epoch),
            )
            discriminator.save(fpath_d)

        print(
            "Time for epoch {} is {} hours".format(
                epoch + 1, (time.time() - start) / 3600
            )
        )

    # guardamos generador y discriminador
    gpath = os.path.join(path_model, gen_model)
    generator.save(gpath)
    dpath = os.path.join(path_model, discr_model)
    discriminator.save(dpath)
    print("Model has been saved")

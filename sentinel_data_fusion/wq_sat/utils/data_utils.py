import os
import numpy as np
from keras.utils import Sequence


def load_data_splits(splits_dir, split_name="train"):
    """
    Load the data arrays from the [train/val/test].txt files.

    Parameters
    ----------
    splits_dir : str
        Absolute path to the image folder.
    split_name : str
        Name of the data split to load
    Returns
    -------
    X : Numpy array of strs
        First colunm: Contains 'absolute_path_to_file' to images.
    """
    if "{}.txt".format(split_name) not in os.listdir(splits_dir):
        raise ValueError(
            "Invalid value for the split_name parameter: there is no `{}.txt` file in the `{}` "
            "directory.".format(split_name, splits_dir)
        )

    # Loading splits
    print("Loading {} data...".format(split_name))
    split = np.genfromtxt(
        os.path.join(splits_dir, "{}.txt".format(split_name)),
        dtype="str",
        delimiter=" ",
    )

    return split


class data_sequence(Sequence):

    def __init__(self, tiles, max_res, batch_size=32, patches_dir=None, shuffle=True):

        if isinstance(tiles, str):
            tiles = [tiles]

        # Create list of inputs and labels
        resolutions = [10, 300]
        self.label_res = np.amax(resolutions)  # resolution of the labels

        inputs, labels = self.tiles_to_samples(tiles, patches_dir, resolutions)
        assert len(inputs) == len(labels)
        assert len(inputs) != 0, (
            "Data generator has length zero. Please provide some data for training/validation."
            "If you don't want to use validation then remove the empty val.txt file."
        )

        self.inputs = inputs
        self.labels = labels
        self.resolutions = [str(res) for res in sorted(resolutions)]
        self.batch_size = np.amin((batch_size, len(inputs)))
        self.shuffle = shuffle
        self.on_epoch_end()

    def tiles_to_samples(self, tiles, patches_dir, resolutions):
        inputs, labels = [], []
        for tilename in tiles:
            tilepath = os.path.join(patches_dir, tilename)

            # Get num_patches
            file_list = os.listdir(tilepath)
            if not file_list:
                continue
            else:
                nums = [int(f.split("_")[1].split(".")[0]) for f in file_list]
                num_patches = np.amax(nums) + 1

            for i in range(num_patches):
                tmp_input = {
                    str(res): os.path.join(tilepath, "input{}_{}.npy".format(res, i))
                    for res in resolutions
                }
                tmp_label = os.path.join(
                    tilepath, "label{}_{}.npy".format(self.label_res, i)
                )

                inputs.append(tmp_input)
                labels.append(tmp_label)

        return inputs, labels

    def __len__(self):
        return int(np.ceil(len(self.inputs) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_idxs = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_X = {res: [] for res in self.resolutions}
        batch_y = []
        for i in batch_idxs:
            batch_y.append(np.load(self.labels[i]))
            for res in self.resolutions:
                batch_X[res].append(np.load(self.inputs[i][res]))

        for k, v in batch_X.items():
            batch_X[k] = np.array(v)

        return batch_X, np.array(batch_y)

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.inputs))
        if self.shuffle:
            np.random.shuffle(self.indexes)


class NPYDataGenerator:
    def __init__(self, tiles, max_res, batch_size=32, patches_dir=None, shuffle=True):
        """
        Inicializa el data generator.

        :param tiles: Lista de imagenes con trozos a entrenar
        :param max_res: resoluciones para entrenar
        :param n_batches: Número de DataFrames a incluir en cada "súper-lote".
        :param patches_dir: carpeta con las carpetas que contienen los npy
        :param shuffle: si sorteamos al acabar la epoca.
        """
        if isinstance(tiles, str):
            tiles = [tiles]

        # Create list of inputs and labels
        resolutions = [10, 300]
        self.label_res = np.amax(resolutions)  # resolution of the labels

        inputs, labels = self.tiles_to_samples(tiles, patches_dir, resolutions)
        assert len(inputs) == len(labels)
        assert len(inputs) != 0, (
            "Data generator has length zero. Please provide some data for training/validation."
            "If you don't want to use validation then remove the empty val.txt file."
        )

        self.inputs = inputs
        self.labels = labels
        self.resolutions = [str(res) for res in sorted(resolutions)]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.current_index = 0

    def tiles_to_samples(self, tiles, patches_dir, resolutions):
        inputs, labels = [], []
        for tilename in tiles:
            print(tilename)
            tilepath = os.path.join(patches_dir, tilename)

            # Get num_patches
            file_list = os.listdir(tilepath)
            if not file_list:
                continue
            else:
                nums = [int(f.split("_")[1].split(".")[0]) for f in file_list]
                num_patches = np.amax(nums) + 1

            for i in range(num_patches):
                tmp_input = {
                    str(res): os.path.join(tilepath, "input{}_{}.npy".format(res, i))
                    for res in resolutions
                }
                tmp_label = os.path.join(
                    tilepath, "label{}_{}.npy".format(self.label_res, i)
                )

                inputs.append(tmp_input)
                labels.append(tmp_label)

        return inputs, labels

    def __len__(self):
        return int(np.ceil(len(self.inputs) / float(self.batch_size)))

    def load_next_dataframe(self):
        """
        Carga el siguiente DataFrame de los cortes de la imagen.
        """
        if self.current_index >= len(self.inputs):
            self.current_index = 0

        sub_inputs = self.inputs[
            self.current_index : self.current_index + self.batch_size
        ]
        if len(sub_inputs) != self.batch_size:
            # Repetir los elementos del array hasta superar el tamaño objetivo
            expanded_array = np.tile(
                sub_inputs, (self.batch_size // len(sub_inputs) + 1)
            )

            # Recortar el array para que tenga exactamente el tamaño deseado
            sub_inputs = expanded_array[: self.batch_size]

        sub_labels = self.labels[
            self.current_index : self.current_index + self.batch_size
        ]
        if len(sub_labels) != self.batch_size:
            # Repetir los elementos del array hasta superar el tamaño objetivo
            expanded_array = np.tile(
                sub_labels, (self.batch_size // len(sub_labels) + 1)
            )

            # Recortar el array para que tenga exactamente el tamaño deseado
            sub_labels = expanded_array[: self.batch_size]

        batch_X = {res: [] for res in self.resolutions}
        batch_y = []
        for i in sub_inputs:
            for res in self.resolutions:
                batch_X[res].append(np.load(i[res]))

        for k, v in batch_X.items():
            batch_X[k] = np.array(v)

        for e in sub_labels:
            batch_y.append(np.load(e))

        self.current_index += self.batch_size

        return batch_X, np.array(batch_y)

    def __iter__(self):
        return self

    def __next__(self):
        for _ in range(self.batch_size):
            X, y = self.load_next_dataframe()
            if X is None:
                if X:
                    break  # Retorna lo que se ha acumulado hasta ahora
                else:
                    raise StopIteration  # No quedan más DataFrames para procesar
        return X, y

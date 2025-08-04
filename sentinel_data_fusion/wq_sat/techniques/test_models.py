"""
This is a file to test the models we have already trained.

Date: February 2019 & June 2023
Author: Ignacio Heredia & María Peña
Email: iheredia@ifca.unican.es & penam@ifca.unican.es
"""

from ..utils import gdal_utils
from osgeo import gdal
import os
import numpy as np
from ..utils.model_utils import super_resolve, load_model


def load_models(model_path, num_layers=32, feature_size=256, drop_and_batch=True):
    """
    Parameters
    --------
    model_path: str
        Path of the model to be loaded
    Returns
    --------
    Keras model
    """
    models = {}
    default_shapes = {"10": (None, None, 4), "300": (None, None, 21)}
    resolutions = [10, 300]
    min_res = min(resolutions)
    for res in resolutions:
        if res == min_res:  # no need to build a model for the minimum resolution
            continue
        input_shape = {
            str(tmp_res): default_shapes[str(tmp_res)]
            for tmp_res in resolutions
            if tmp_res <= res
        }
        models[res] = load_model(
            input_shape=input_shape,
            model_path=model_path,
            num_layers=num_layers,
            feature_size=feature_size,
            drop_and_batch=drop_and_batch,
        )
    return models


def test(
    tile_name,
    model_path,
    data_path,
    output_path,
    output_name,
    num_layers=32,
    feature_size=256,
    downsamples=False,
    max_res=300,
    output_file_format="GTiff",
    drop_and_batch=True,
):
    """
    Parameters
    --------
    tile_name: str
        Name of the image file
    model_path: str
        Path of the model to test
    downsamples: bool
        True if images to be SR are from low-scale
    max_res: int
        Maximum resolution considered
    data_path: str
        Directory of input images
    output_path: str
        Path where SR image is saved
    output_file_format: str
        Format of the output file
    --------
    Keras model
    """
    # Check resolutions
    sat_resolutions = list([10, 300])
    min_res = min(sat_resolutions)
    if max_res is None:  # we super-resolve for all possible resolutions
        max_res = max(sat_resolutions)
        sr_resolutions = [
            res for res in sat_resolutions if res != min_res
        ]  # en S3 es 300
    else:  # si tenemos una resolución máxima sr_resolutions toma su valor
        assert (
            max_res in sat_resolutions
        ), "The selected resolution is not an available choice"
        assert (
            max_res != min_res
        ), "The super-resolution must be larger than the smaller resolution"
        sr_resolutions = [max_res]

    models = load_models(
        model_path,
        num_layers=num_layers,
        feature_size=feature_size,
        drop_and_batch=drop_and_batch,
    )

    region_path = os.path.join(data_path, tile_name)
    files = os.listdir(region_path)
    tile_path_s2 = os.path.join(
        region_path,
        next(
            (file for file in files if file.startswith("S2") and file.endswith("tif")),
            None,
        ),
    )
    tile_path_s3 = os.path.join(
        region_path,
        next(
            (file for file in files if file.startswith("S3") and file.endswith("tif")),
            None,
        ),
    )

    # cargamos el array del fichero de sentinel-2
    data_bands = {}
    ds_s2 = gdal.Open(tile_path_s2)
    coord_geotransform = ds_s2.GetGeoTransform()
    coord_geoprojection = ds_s2.GetProjection()
    data_bands[10] = ds_s2.ReadAsArray()
    data_bands[10] = np.moveaxis(data_bands[10], source=0, destination=2)
    ds_s2 = None

    # cargamos el array del fichero de sentinel-3
    data_bands[300] = []
    ds_s3 = gdal.Open(tile_path_s3)
    data_bands[300] = ds_s3.ReadAsArray()
    data_bands[300] = np.moveaxis(data_bands[300], source=0, destination=2)
    ds_s3 = None

    # Check image
    if not np.any(data_bands[min_res]):
        raise Exception("The selected region is empty.")

    # Borders and patch_sizes for inference
    patch_sizes = {300: 120}
    borders = {300: 30}

    # Perform super-resolution
    # inicializa sr_bands con las resoluciones 20 y 60 como llaves
    sr_bands = {res: None for res in sr_resolutions}
    for res in sr_bands.keys():
        print("Super resolving {}m ...".format(res))
        # si la resolucion considerada en data_bands está por debajo o es igual a 300 se añade al diccionario tmp_bands
        tmp_bands = {
            tmp_res: bands for tmp_res, bands in data_bands.items() if tmp_res <= res
        }
        min_side = min(
            tmp_bands[res].shape[:2]
        )  # calculo el menor entre width y height de la imagen
        tmp_patchsize = min(
            patch_sizes[res], min_side
        )  # y el mínimo entre el lado más pequeño y el patch para esa resolución (no sé bien para qué se hace)
        tmp_patchsize = patch_sizes[
            res
        ]  # con esto se puede trabajar con imágenes más pequeñas sin que dé fallo
        sr_bands[res] = super_resolve(
            data_bands=tmp_bands,
            model=models[res],
            downsamples=downsamples,
            patch_size=tmp_patchsize,
            border=borders[res],
        )

    # escribimos diccionarios con las bandas y las descripciones
    res_to_bands = {
        10: ["B4", "B3", "B2", "B8"],
        300: [
            "Oa01",
            "Oa02",
            "Oa03",
            "Oa04",
            "Oa05",
            "Oa06",
            "Oa07",
            "Oa08",
            "Oa09",
            "Oa10",
            "Oa11",
            "Oa12",
            "Oa13",
            "Oa14",
            "Oa15",
            "Oa16",
            "Oa17",
            "Oa18",
            "Oa19",
            "Oa20",
            "Oa21",
        ],
    }

    band_desc = {
        "B4": "B4 (665 nm)",
        "B3": "B3 (560 nm)",
        "B2": "B2 (490 nm)",
        "B8": "B8 (842 nm)",
        "Oa01": "Oa01",
        "Oa02": "Oa02",
        "Oa03": "Oa03",
        "Oa04": "Oa04",
        "Oa05": "Oa05",
        "Oa06": "Oa06",
        "Oa07": "Oa07",
        "Oa08": "Oa08",
        "Oa09": "Oa09",
        "Oa10": "Oa10",
        "Oa11": "Oa11",
        "Oa12": "Oa12",
        "Oa13": "Oa13",
        "Oa14": "Oa14",
        "Oa15": "Oa15",
        "Oa16": "Oa16",
        "Oa17": "Oa17",
        "Oa18": "Oa18",
        "Oa19": "Oa19",
        "Oa20": "Oa20",
        "Oa21": "Oa21",
    }

    # Join the non-empty super resolved bands
    sr, validated_sr_bands = [], []
    for k, v in sr_bands.items():
        sr.append(v)
        validated_sr_bands += res_to_bands[k]
    sr = np.concatenate(sr, axis=2)

    # Create the lists of output variables to save
    output_bands, output_desc, output_shortnames = [], [], []

    for bi, bn in enumerate(validated_sr_bands):
        output_bands.append(sr[:, :, bi])
        output_desc.append("SR" + band_desc[bn])
        output_shortnames.append("SR" + bn)

    # guardamos la imagen SR
    gdal_utils.save_gdal_test(
        output_path=os.path.join(output_path, output_name),
        bands=output_bands,
        descriptions=output_desc,
        geotransform=coord_geotransform,
        geoprojection=coord_geoprojection,
        file_format=output_file_format,
    )

    print("Prediction on file from {} has been done \n".format(tile_name))

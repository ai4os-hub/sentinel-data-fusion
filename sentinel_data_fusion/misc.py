"""
This file gathers some functions that have proven to be useful
across projects. They are not strictly need for integration
but users might want nevertheless to take advantage from them.
"""

from sentinel_data_fusion import config

from functools import wraps
from multiprocessing import Process
import subprocess  # nosec
import warnings

from aiohttp.web import HTTPBadRequest

import os
import dateutil.relativedelta
from datetime import datetime
import numpy as np

from osgeo import gdal
from skimage.transform import resize
import glob


from sentinel_data_fusion.wq_sat.satellites.download_sentinel import download as dl_st
from sentinel_data_fusion.wq_sat.utils import preprocessing_utils as preproc_utils
from sentinel_data_fusion.wq_sat.satellites.sentinel2 import S2Reader as st2_bands
from sentinel_data_fusion.wq_sat.satellites.sentinel3 import S3Reader as st3_bands
from sentinel_data_fusion.wq_sat.techniques.test_models import test as test


def _catch_error(f):
    """
    Decorate API functions to return an error as HTTPBadRequest,
    in case it fails.
    """

    @wraps(f)
    def wrap(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            # raise HTTPBadRequest(reason=e)
            print("HTTPBadRequest")

    return wrap


def _fields_to_dict(fields_in):
    """
    Function to convert marshmallow fields to dict()
    """
    dict_out = {}
    for k, v in fields_in.items():
        param = {}
        param["default"] = v.missing
        param["type"] = type(v.missing)
        param["required"] = getattr(v, "required", False)

        v_help = v.metadata["description"]
        if "enum" in v.metadata.keys():
            v_help = f"{v_help}. Choices: {v.metadata['enum']}"
        param["help"] = v_help

        dict_out[k] = param

    return dict_out


def mount_nextcloud(frompath, topath):
    """
    Mount a NextCloud folder in your local machine or viceversa.

    Example of usage:
        mount_nextcloud('rshare:/data/images', 'my_local_image_path')

    Parameters
    ==========
    * frompath: str, pathlib.Path
        Source folder to be copied
    * topath: str, pathlib.Path
        Destination folder
    """
    command = ["rclone", "copy", f"{frompath}", f"{topath}"]
    result = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE  # nosec
    )
    output, error = result.communicate()
    if error:
        warnings.warn(f"Error while mounting NextCloud: {error}")
    return output, error


def launch_cmd(logdir, port):
    subprocess.call(
        [
            "tensorboard",  # nosec
            "--logdir",
            f"{logdir}",
            "--port",
            f"{port}",
            "--host",
            "0.0.0.0",
        ]
    )


def launch_tensorboard(logdir, port=6006):
    """
    Run Tensorboard on a separate Process on behalf of the user

    Parameters
    ==========
    * logdir: str, pathlib.Path
        Folder path to tensorboard logs.
    * port: int
        Port to use for the monitoring webserver.
    """
    subprocess.run(  # nosec
        # kill any other process in that port
        ["fuser", "-k", f"{port}/tcp"]  # nosec
    )
    p = Process(target=launch_cmd, args=(logdir, port), daemon=True)
    p.start()


# utils.py

from datetime import date, datetime
from typing import Dict, Any


def predict_for_bbox(
    username: str,
    password: str,
    lon_min: float,
    lat_min: float,
    lon_max: float,
    lat_max: float,
    start_date: str,
    end_date: str,
    output_name: str,
) -> Dict[str, Any]:
    """
    Ejecuta la predicción para el bounding box y rango de fechas indicados.

    Parámetros:
    -----------
    lon_min, lat_min, lon_max, lat_max : float
        Coordenadas del rectángulo [minX, minY, maxX, maxY]
    start_date, end_date : datetime.date
        Fecha de inicio y fin del periodo de interés.

    Devuelve:
    ---------
    Un diccionario con la predicción y metadatos, por ejemplo:
      {
        "mean_value": 0.42,
        "std_value": 0.13,
        "n_observations": 12,
        ...
      }
    """

    # 1) Validar fechas
    if start_date > end_date:
        raise ValueError(f"start_date ({start_date}) must be <= end_date ({end_date})")

    # 2) Prepara parámetros
    params = {
        "bbox": {
            "lon_min": lon_min,
            "lat_min": lat_min,
            "lon_max": lon_max,
            "lat_max": lat_max,
        },
        "period": {"from": start_date, "to": end_date},
    }

    # 3) Llamada a tu motor de predicción real
    # (Reemplaza esta función por la tuya, por ejemplo:
    #   from sentinel_data_fusion.core import run_model_on_bbox
    #   result = run_model_on_bbox(lon_min, lat_min, lon_max, lat_max, start_date, end_date)
    # )
    downloading(
        username,
        password,
        output_name,
        {"N": lon_min, "W": lat_min, "S": lon_max, "E": lat_max},
        start_date,
        end_date,
    )
    saving_tifs(output_name, {"N": lon_min, "W": lat_min, "S": lon_max, "E": lat_max})
    output_file = super_resolving(output_name)
    return output_file


def downloading(username, password, file_name, coordinates, start_date, end_date):
    """
    Parameters
    --------
    file_name: str
        Name of the final output
    coordinates: dict
        Dict with the coordinates of the desired area. Example: {N: 43.5, W: -4.13, S: 43.39, E: -3.43}
    """
    # descargamos el número lim_downloads_s2 de ficheros de sentinel-2 con las especificaciones que queremos
    lim_downloads_s2 = 1
    recov_images_s2_total = dict()
    recov_images_s3_total = dict()
    output_path = os.path.join("/srv")

    print(start_date)

    # generamos los objetos con los requerimientos del TFM y los descargamos
    s2 = dl_st(
        inidate=datetime.fromisoformat(start_date).isoformat(),
        enddate=datetime.fromisoformat(end_date).isoformat(),
        producttype="S2MSI1C",
        region_name=file_name,
        lim_downloads=lim_downloads_s2,
        platform="Sentinel-2",
        cloud=1,
        coordinates=coordinates,
        output_path=output_path,
        username=username,
        password=password,
    )

    products = s2.search()
    if len(products) > lim_downloads_s2:
        products = products[:lim_downloads_s2]
    scenes_s2, recov_images_s2 = s2.download()
    recov_images_s2_total.update(recov_images_s2)

    count_res = 0

    # Para cada uno de los productos encontrados busco un producto de Sentinel-3 que incluyan el mismo área
    # para el mismo día (+-1)
    for prod in products:
        count_res += 1

        # buscamos que la fecha de colección del producto diste en, como mucho, 1 día respecto al de Sentinel-2
        inidate = (
            "".join(
                str(
                    datetime.fromisoformat(prod["OriginDate"][:10])
                    + dateutil.relativedelta.relativedelta(days=-1)
                ).split(" ")[0]
            )
            + "T00:00:00Z"
        )
        enddate = (
            "".join(
                str(
                    datetime.fromisoformat(prod["OriginDate"][:10])
                    + dateutil.relativedelta.relativedelta(days=+1)
                ).split(" ")[0]
            )
            + "T00:00:00Z"
        )

        footprint = prod["Footprint"][20:-1]

        s3 = dl_st(
            inidate=inidate,
            enddate=enddate,
            producttype="OL_1_EFR___",
            region_name=file_name,
            lim_downloads=1,
            platform="Sentinel-3",
            footprint=footprint,
            output_path=output_path,
        )

        scenes_s3, recov_images_s3 = s3.download()
        recov_images_s3_total.update(recov_images_s3)


def saving_tifs(file_name, coordinates):

    # DEFINIMOS EL PATH DE SALIDA
    output_path = os.path.join("/srv", file_name)

    # CORRECCIONES Y CONVERSIÓN A TIF DEL FICHERO DE SENTINEL-3
    file = glob.glob(os.path.join("/srv", file_name, "*SEN3"))[0]
    reader = st3_bands(file, output_path, atcor=True)
    reader.read_bands()

    # CORRECCIONES Y CONVERSIÓN A TIF DEL FICHERO DE SENTINEL-2
    file = glob.glob(os.path.join("/srv", file_name, "*.SAFE"))[0]
    reader = st2_bands(file, output_path, atcor=True)
    reader.read_bands()

    # DEFINIMOS LOS PATHS
    files = os.listdir(output_path)
    tif_files = [file for file in files if file.endswith("tif")]
    # Encontrar los archivos S2 y S3
    s2_file = next((file for file in tif_files if file.startswith("S2")), None)
    s3_file = next((file for file in tif_files if file.startswith("S3")), None)
    # Obtener las rutas completas de los archivos S2 y S3
    s2_path = os.path.join(output_path, s2_file)
    s3_path = os.path.join(output_path, s3_file)

    # PRIMER CROP DE SENTINEL-3 SEGÚN LAS COORDENADAS QUE SE ESPECIFICAN DE ENTRADA
    preproc_utils.crop_tif(s3_path, output_path, coordinates)

    # RESIZE DE SENTINEL-3 Y CORTE CONJUNTO DE SENTINEL-2 Y 3
    preproc_utils.resize_tif_s3(s2_path, s3_path, output_path)
    intersection = preproc_utils.calculate_intersection(s2_path, s3_path)
    # llamo a las funciones generadas para recortar las imágenes
    preproc_utils.crop_tif(s2_path, output_path, intersection)
    preproc_utils.crop_tif(s3_path, output_path, intersection)

    # COMPROBACIÓN DE COINCIDENCIA DE COORDENADAS
    print("Se comprueba si las coordenadas de Sentinel 2 y 3 coinciden:")
    ds = gdal.Open(s2_path)
    width, height, gt = ds.RasterXSize, ds.RasterYSize, ds.GetGeoTransform()
    minx, miny, maxx, maxy = gt[0], gt[3] + height * gt[5], gt[0] + width * gt[1], gt[3]
    ds3 = gdal.Open(s3_path)
    width3, height3, gt3 = ds3.RasterXSize, ds3.RasterYSize, ds3.GetGeoTransform()
    minx3, miny3, maxx3, maxy3 = (
        gt3[0],
        gt3[3] + height3 * gt3[5],
        gt3[0] + width3 * gt3[1],
        gt3[3],
    )
    ds, ds3 = None, None
    while (
        np.round(minx, 8) != np.round(minx3, 8)
        or np.round(miny, 8) != np.round(miny3, 8)
        or np.round(maxx, 8) != np.round(maxx3, 8)
        or np.round(maxy, 8) != np.round(maxy3, 8)
    ):
        print("Las coordenadas de ambas misiones aún no coinciden")
        # llamo a la función que calcula la intersección entre coordenadas de ambas imágenes
        intersection = preproc_utils.calculate_intersection(s2_path, s3_path)
        # llamo a las funciones generadas para recortar las imágenes
        preproc_utils.crop_tif(s2_path, output_path, intersection)
        preproc_utils.crop_tif(s3_path, output_path, intersection)
        ds = gdal.Open(s2_path)
        width, height, gt = ds.RasterXSize, ds.RasterYSize, ds.GetGeoTransform()
        minx, miny, maxx, maxy = (
            gt[0],
            gt[3] + height * gt[5],
            gt[0] + width * gt[1],
            gt[3],
        )
        ds3 = gdal.Open(s3_path)
        width3, height3, gt3 = ds3.RasterXSize, ds3.RasterYSize, ds3.GetGeoTransform()
        minx3, miny3, maxx3, maxy3 = (
            gt3[0],
            gt3[3] + height3 * gt3[5],
            gt3[0] + width3 * gt3[1],
            gt3[3],
        )
        ds, ds3 = None, None
    print(
        "Las coordenadas de ambas misiones coinciden: W:",
        minx,
        "S:",
        miny,
        "E:",
        maxx,
        "N:",
        maxy,
    )


def super_resolving(file_name):

    output_path = os.path.join("/srv", file_name)

    test(
        file_name,
        model_path="/srv/sentinel-data-fusion/models/cnn_model_60epochs_exactPatches_35tiles_32lay_256fm_normmeanvar.h5",
        num_layers=32,
        feature_size=256,
        downsamples=False,
        max_res=300,
        data_path="/srv",
        drop_and_batch=False,
        output_path=output_path,
        output_name=file_name + ".tiff",
        output_file_format="GTiff",
    )
    return output_path + "/" + file_name + ".tiff"

# -*- coding: utf-8 -*-
"""
Functions to integrate your model with the DEEPaaS API.
It's usually good practice to keep this file minimal, only performing
the interfacing tasks. In this way you don't mix your true code with
DEEPaaS code and everything is more modular. That is, if you need to write
the predict() function in api.py, you would import your true predict function
and call it from here (with some processing / postprocessing in between
if needed).
For example:

    import mycustomfile
bbo
    def predict(**kwargs):
        args = preprocess(kwargs)
        resp = mycustomfile.predict(args)
        resp = postprocess(resp)
        return resp

To start populating this file, take a look at the docs [1] and at
an exemplar module [2].

[1]: https://docs.ai4os.eu/
[2]: https://github.com/ai4os-hub/ai4os-demo-app
"""

from pathlib import Path
import logging

from sentinel_data_fusion import config
from sentinel_data_fusion.misc import _catch_error
import sentinel_data_fusion.misc as misc
from webargs import fields
from marshmallow import validate
import re

# set up logging
logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)

BASE_DIR = Path(__file__).resolve().parents[1]


@_catch_error
def get_metadata():
    """Returns a dictionary containing metadata information about the module.
       DO NOT REMOVE - All modules should have a get_metadata() function

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:  # Call your AI model metadata() method
        logger.info("Collecting metadata from: %s", config.API_NAME)
        metadata = config.PROJECT_METADATA
        # TODO: Add dynamic metadata collection here
        logger.debug("Package model metadata: %s", metadata)
        names = re.findall(r'"([^"]+)"', metadata["author"])
        if names:
            metadata["author"] = ", ".join(names)
        else:
            metadata["author"] = metadata["author"].strip('"')
        return metadata
    except Exception as err:
        logger.error("Error collecting metadata: %s", err, exc_info=True)
        raise  # Reraise the exception after log


# def warm():
#     pass
#
#
def get_predict_args():
    """
    **Descripción general de la operación**

    Aquí puedes explicar **qué hace** este endpoint,
    qué retorno esperas, advertencias, links de ayuda, etc.

    Luego vienen los parámetros:
    - `bbox` (requerido): …
    - `start_date`, `end_date`: …

    Según el header `Accept` se devolverá:
    - un TIFF (image/tiff)
    - o un JSON (application/json).
    """
    return {
        "copernicus_user": fields.String(
            required=True,
            validate=validate.Length(min=1),
            metadata={
                "description": "User for downloading COPERNICUS data.",
            },
        ),
        "copernicus_password": fields.String(
            required=True,
            validate=validate.Length(min=1),
            load_only=True,
            metadata={
                "description": "Password for downloading COPERNICUS data.",
                "format": "password",
            },
        ),
        "bbox(N,W,S,E)": fields.List(
            fields.Float,
            required=False,
            missing=[43.5, -4.13, 43.39, -3.43],
            location="json",
            description="Bounding box: [N, W, S, E] - Decimal. To get the coordinates, you can use services like https://boundingbox.klokantech.com/",
            validate=validate.Length(equal=4),
        ),
        "start_date(YYYY-MM-DD)": fields.Str(
            required=False,
            missing="2024-01-01",
            location="json",
            description="Start date (YYYY-MM-DD)",
        ),
        "end_date(YYYY-MM-DD)": fields.Str(
            required=False,
            missing="2025-07-01",
            location="json",
            description="End date (YYYY-MM-DD)",
        ),
        "output_name": fields.Str(
            required=False,
            missing="output_file",
            location="json",
            description="Name of file to fusion",
        ),
        "accept": fields.Str(
            required=False,
            missing="image/tiff",
            location="headers",
            description="Media type acceptable for the response",
            validate=validate.OneOf(
                ["application/json", "image/tiff", "application/zip"]
            ),
        ),
    }


def predict(**kwargs):
    """
    Ejecuta la inferencia y devuelve o bien JSON o bien los bytes del TIFF
    según el header 'accept'.
    """
    username = kwargs["copernicus_user"]
    password = kwargs["copernicus_password"]
    lon_min, lat_min, lon_max, lat_max = kwargs["bbox(N,W,S,E)"]
    start_date = kwargs["start_date(YYYY-MM-DD)"]
    end_date = kwargs["end_date(YYYY-MM-DD)"]
    output_name = kwargs["output_name"]
    accept = kwargs.get("accept")

    # Llama a tu función que crea el .tif y devuelve su ruta
    tif_path = misc.predict_for_bbox(
        username=username,
        password=password,
        lon_min=lon_min,
        lat_min=lat_min,
        lon_max=lon_max,
        lat_max=lat_max,
        start_date=start_date,
        end_date=end_date,
        output_name=output_name,
    )

    if accept == "image/tiff":
        try:
            return open(tif_path, "rb")
        except FileNotFoundError:
            raise Exception(f"File not found: {tif_path}")

    # Fallback a JSON si se pidiera application/json
    return {
        "bbox": {
            "lon_min": lon_min,
            "lat_min": lat_min,
            "lon_max": lon_max,
            "lat_max": lat_max,
        },
        "period": {"from": start_date.isoformat(), "to": end_date.isoformat()},
        "tif_path": tif_path,
    }


#
#
# @_catch_error
# def predict(**kwargs):
#     return None
#
#
# def get_train_args():
#     return {}
#
#
# def train(**kwargs):
#     return None

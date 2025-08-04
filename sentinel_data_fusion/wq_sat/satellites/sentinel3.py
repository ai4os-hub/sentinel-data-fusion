import os
import warnings
import numpy as np
import rioxarray
import xarray as xr
import rasterio
from rasterio.control import GroundControlPoint
from scipy.interpolate import RectBivariateSpline
from ..satellites import atcor


class S3Reader:
    """
    Class to read data from S3
    """

    def __init__(self, tile_path, output_path, bands=None, atcor=True):
        """
        Initialize the class with the bucket and key
        """

        warnings.filterwarnings("ignore", category=UserWarning, module="rioxarray._io")

        self.tile_path = tile_path.rstrip("/") + "/"  # Evitar doble barra
        self.output_path = output_path
        self.atcor = atcor

        if bands is None:
            bands = [
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
            ]
        self.bands = bands

    def get_tr(self):
        """
        Read the latitude and longitude from S3
        """

        lat = rioxarray.open_rasterio(
            f"netcdf:{self.tile_path}geo_coordinates.nc:latitude"
        )
        lon = rioxarray.open_rasterio(
            f"netcdf:{self.tile_path}geo_coordinates.nc:longitude"
        )

        lat_scale = getattr(lat, "scale_factor", 1)
        lon_scale = getattr(lon, "scale_factor", 1)

        lon_corrected = lon.data[0] * lon_scale
        lat_corrected = lat.data[0] * lat_scale

        tr = rasterio.transform.from_bounds(
            west=np.min(lon_corrected),
            south=np.min(lat_corrected),
            east=np.max(lon_corrected),
            north=np.max(lat_corrected),
            width=lat.x.size,
            height=lat.y.size,
        )

        return lat_corrected, lon_corrected, tr

    def get_SunAngles(self, width, height):
        """
        Read the Sun angles from S3
        """

        file_path = os.path.join(self.tile_path, "tie_geometries.nc")
        ds = xr.open_dataset(file_path)

        sza = ds["SZA"].values
        saa = ds["SAA"].values

        tie_x = np.linspace(0, width, sza.shape[1])
        tie_y = np.linspace(0, height, sza.shape[0])

        full_x = np.linspace(0, width, width)
        full_y = np.linspace(0, height, height)

        sza_interp = RectBivariateSpline(tie_y, tie_x, sza)(full_y, full_x)
        saa_interp = RectBivariateSpline(tie_y, tie_x, saa)(full_y, full_x)

        return sza_interp, saa_interp

    def read_bands(self):
        """
        Read the data from S3
        """

        valid_bands = [
            file
            for file in os.listdir(self.tile_path)
            if file.endswith("radiance.nc")
            and any(file.startswith(band) for band in self.bands)
        ]

        if not valid_bands:
            raise ValueError("No valid bands found in the Sentinel-3 tile.")

        valid_bands.sort(
            key=lambda x: self.bands.index(
                next(band for band in self.bands if x.startswith(band))
            )
        )

        lat, lon, tr = self.get_tr()
        ds_list = []

        for b in valid_bands:
            banda = b.split(".")[0]
            A = rioxarray.open_rasterio(f"netcdf:{self.tile_path}{banda}.nc:{banda}")
            arr = A.data[0]
            if self.atcor:
                scale_factor = getattr(A, "scale_factor", 1)
                sza, saa = self.get_SunAngles(A.sizes["x"], A.sizes["y"])
                arr = atcor.Atcor(arr, sza, saa, scale_factor).apply_all_corrections()
            ds_list.append(arr)
        arr_bands = np.array(ds_list)
        del ds_list
        A = xr.DataArray(
            arr_bands,
            coords=[np.array(range(1, len(valid_bands) + 1)), A.y.data, A.x.data],
            dims=A.dims,
        )
        A.rio.write_crs("EPSG:4326", inplace=True)
        A.rio.write_transform(transform=tr, inplace=True)
        nof_gcp_x = np.arange(0, A.x.size, 100)
        nof_gcp_y = np.arange(0, A.y.size, 100)
        gcps = []
        id = 0
        for x in nof_gcp_x:
            for y in nof_gcp_y:
                gcps.append(
                    GroundControlPoint(
                        row=int(y),
                        col=int(x),
                        x=float(lon[y, x]),
                        y=float(lat[y, x]),
                        z=0.0,
                        id=id,
                    )
                )
                id += 1
        rasterio.transform.from_gcps(gcps)
        A = A.rio.reproject(dst_crs="EPSG:4326", gcps=gcps, **{"SRC_METHOD": "GCP_TPS"})
        tile_id = self.tile_path.split("/")[-2][:-5]
        output_file = "%s/%s.tif" % (self.output_path, tile_id)
        A.rio.to_raster(output_file, recalc_transform=False)
        print("File is saved in {}".format(output_file))

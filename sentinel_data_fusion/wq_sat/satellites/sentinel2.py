import os
import glob
import defusedxml.ElementTree as ET
import rasterio
import rasterio.merge
from ..satellites import atcor
from osgeo import gdal


class S2Reader:
    def __init__(
        self, tile_path, output_path, bands=["B04", "B03", "B02", "B08"], atcor=True
    ):

        self.tile_path = tile_path
        self.output_path = output_path
        self.atcor = atcor
        # Default to all available bands if none are specified
        self.bands = bands or [
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "B06",
            "B07",
            "B08",
            "B8A",
            "B09",
            "B11",
            "B12",
        ]

    def get_SunAngles(self):

        # Locate the MTD_TL.xml file inside the GRANULE subdirectory
        granule_path = os.path.join(self.tile_path, "GRANULE")
        if not os.path.exists(granule_path):
            raise FileNotFoundError(f"Granule directory not found: {granule_path}")

        granule_dirs = [
            d
            for d in os.listdir(granule_path)
            if os.path.isdir(os.path.join(granule_path, d))
        ]
        if not granule_dirs:
            raise FileNotFoundError(
                "No granule subdirectory found inside GRANULE folder."
            )

        metadata_path = os.path.join(granule_path, granule_dirs[0], "MTD_TL.xml")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        # Parse the XML
        tree = ET.safe_parse(metadata_path)
        root = tree.getroot()

        # Extract solar angle from the correct file
        sun_angle_node = root.find(".//Mean_Sun_Angle")
        if sun_angle_node is not None:
            sza = (float(sun_angle_node.find("ZENITH_ANGLE").text),)
            saa = float(sun_angle_node.find("AZIMUTH_ANGLE").text)

        return sza, saa

    def read_bands(self):

        try:
            # Locate the IMG_DATA folder within the Sentinel-2 tile structure
            granule_path = glob.glob(
                os.path.join(self.tile_path, "GRANULE", "*", "IMG_DATA")
            )[0]
            jp2_files = {
                os.path.basename(f).split("_")[-1].split(".")[0]: f
                for f in glob.glob(os.path.join(granule_path, "*.jp2"))
            }
        except IndexError:
            raise ValueError("Granule path not found in the Sentinel-2 tile directory.")

        valid_bands = [b for b in self.bands if b in jp2_files]
        valid_bands.sort(
            key=lambda x: self.bands.index(
                next(b for b in self.bands if x.startswith(b))
            )
        )
        if not valid_bands:
            raise ValueError("No valid bands found in the Sentinel-2 tile.")

        resolution_bands = {
            "10": ["B04", "B03", "B02", "B08"],
            "20": ["B05", "B06", "B07", "B8A", "B11", "B12"],
            "60": ["B01", "B09", "B10"],
        }

        arr_bands, metadata = {}, {}
        if self.atcor:
            sza, saa = self.get_SunAngles()

        for res, bands in resolution_bands.items():
            arr_bands[res] = {}
            bands_list = []
            band_arrays = []

            for band in bands:
                if band not in valid_bands:
                    continue
                try:
                    with rasterio.open(jp2_files[band]) as src:
                        window, new_bounds = None, None
                        data = src.read(1, window=window)

                        if self.atcor:
                            data = atcor.Atcor(data, sza, saa).apply_all_corrections()

                        arr_bands[res][band] = data
                        bands_list.append(band)
                        band_arrays.append(data)

                        if res not in metadata:
                            meta = src.meta.copy()
                            if window:
                                meta.update(
                                    {
                                        "transform": src.window_transform(window),
                                        "width": data.shape[1],
                                        "height": data.shape[0],
                                    }
                                )
                            meta.update(
                                {
                                    "count": 0,  # Placeholder; updated after
                                    "dtype": data.dtype,
                                    "bands": [],
                                    "driver": "GTiff",
                                    "bounds": new_bounds if new_bounds else src.bounds,
                                }
                            )
                            metadata[res] = meta

                except Exception as e:
                    raise RuntimeError(
                        f"Error reading band {band} at {res}m resolution: {e}"
                    )

            # Save the TIFF if bands were read
            tile_id = self.tile_path.split("/")[-1][:-5]
            output_file = "%s/%s.tif" % (self.output_path, tile_id)
            if bands_list:
                meta = metadata[res]
                meta.update({"count": len(bands_list), "bands": bands_list})

                with rasterio.open(output_file, "w", **meta) as dst:
                    for idx, band_data in enumerate(band_arrays, start=1):
                        dst.write(band_data, idx)
                print(f"File is saved in {output_file}")

        dataset = gdal.Open(output_file)
        gdal.Warp(
            output_file, dataset, dstSRS="EPSG:4326", srcNodata=0, dstNodata=-9999
        )

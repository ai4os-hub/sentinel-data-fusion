"""
This is a file to crop tif files

Date: June 2023
Author: María Peña
Email: penam@ifca.unican.es
"""

import rasterio
import os
from .. import config
from osgeo import gdal
from rasterio.windows import Window, from_bounds, transform as window_transform
import math
from skimage.transform import resize
from rasterio.enums import Resampling
from shapely.geometry import box
import numpy as np

def crop_tif(file, output_path, coord): 
    """
    Crop file tif to required coordinates
    ----------
    Parameters
    file: str
        Name of the file to be cropped
    coord: dict
        Dictionary with 'N','E','S','W' as keys, and coordinates as values
    satellite: str
        Name of the file's mission
    """    
    output_file = os.path.join(output_path, file.split('/')[-1])
    
    # recorto la imagen según las especificaciones y lo guardo en el fichero de salida
    with rasterio.open(file) as src:
        window = from_bounds(coord['W'], coord['S'], coord['E'], coord['N'], src.transform)
        kwargs = src.meta.copy()
        kwargs.update({
            'height': window.height,
            'width': window.width,
            'transform': rasterio.windows.transform(window, src.transform)})

        with rasterio.open(output_file, 'w', **kwargs) as dst:
            dst.write(src.read(window=window))    
            print('File has been cropped in {}'.format(output_file))
            

def find_largest_non_nan_rectangle(mask): 
    """
    Encuentra el rectángulo más grande en la matriz booleana "mask" 
    (True donde el pixel es válido, False donde es NaN) sin valores False.

    Parámetros:
    - mask: numpy.ndarray 2D booleana.

    Retorna:
    - coords: diccionario con 'top', 'left', 'height', 'width' del rectángulo.
    """
    rows, cols = mask.shape
    heights = np.zeros(cols, dtype=int)

    max_area = 0
    best_rect = {'top': 0, 'left': 0, 'height': 0, 'width': 0}

    for r in range(rows):
        # Actualiza las alturas: si mask[r, c] es True, sumar 1; si es False, reiniciar a 0
        heights = heights + 1
        heights[~mask[r]] = 0

        # Encontrar la mayor área en el histograma “heights”
        stack = []
        for i in range(cols + 1):
            current_height = heights[i] if i < cols else 0
            while stack and current_height < heights[stack[-1]]:
                top_idx = stack.pop()
                h = heights[top_idx]
                w = i if not stack else (i - stack[-1] - 1)
                area = h * w
                if area > max_area:
                    max_area = area
                    top_row = r - h + 1
                    left_col = i - w
                    best_rect = {
                        'top': top_row,
                        'left': left_col,
                        'height': h,
                        'width': w
                    }
            stack.append(i)

    return best_rect



            
def calculate_intersection(folder_crop_s2, folder_crop_s3):  
    """
    Calculate intersection between two files
    ----------
    Parameters
    folder_crop_s2: str
        Name of the Sentinel-2 file to be intersected
    folder_crop_s3: str
        Name of the Sentinel-3 file to be intersected
    remove_nans: bool
        If nans are considered when cropping or not
    """ 
    cropped_results = []
    
    for img in [folder_crop_s2, folder_crop_s3]:
        with rasterio.open(img) as src:
            data = src.read().astype(float)
            data[data == -9999] = np.nan
            profile = src.profile
    
            mask_2d = ~np.any(np.isnan(data), axis=0)
            coords = find_largest_non_nan_rectangle(mask_2d)
    
            row_offset = coords['top']
            col_offset = coords['left']
            height = coords['height']
            width = coords['width']
    
            window = Window(col_offset, row_offset, width, height)
            new_transform = window_transform(window, src.transform)
            bounds = rasterio.windows.bounds(window, src.transform)
    
            cropped_results.append({
                "input_path": img,
                "window": window,
                "transform": new_transform,
                "profile": profile,
                "bounds": bounds
            })

    poly_1 = box(*cropped_results[0]["bounds"])
    poly_2 = box(*cropped_results[1]["bounds"])
    intersection = poly_1.intersection(poly_2)
    interb = intersection.bounds
    
    # hago la intersección de las coordenadas de los bordes de ambas imágenes para recortarlas
    intersection = {'W': interb[0], 
                    'N': interb[3],
                    'E': interb[2],
                    'S': interb[1]}

    return intersection
    

    
def resize_tif_s3(folder_crop_s2, folder_crop_s3, output_path): 
    """
    Resize of Sentinel-3 image such that a square of 30-pixels-size of Sentinel-2 image
    corresponds to one pixel of Sentinel-3 one
    ----------
    Parameters
    folder_crop_s2: str
        Name of the Sentinel-2 file 
    folder_crop_s3: str
        Name of the Sentinel-3 file to be resized
    """     
    
    # obtenemos la imagen de s2
    ds = gdal.Open(folder_crop_s2)
    gt1 = list(ds.GetGeoTransform())
    
    # obtenemos la imagen de s3
    ds3 = gdal.Open(folder_crop_s3)
    gt2 = list(ds3.GetGeoTransform())  
    
    # calculamos el factor de conversión de los píxeles
    factor = (gt1[1]*30)/gt2[1]

    output_file = os.path.join(output_path, folder_crop_s3.split('/')[-1])
    
    # creamos el nuevo fichero donde irá la imagen tras el resize
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_file,    
                            round(ds3.RasterXSize/factor), # columnas
                            round(ds3.RasterYSize/factor), # filas
                            21, gdal.GDT_Float64)
    gt2[1]=gt2[1]*factor
    gt2[5]=gt2[5]*factor
    outdata.SetGeoTransform(tuple(gt2))##sets same geotransform as input
    outdata.SetProjection(ds3.GetProjection())##sets same projection as input
    
    # hacemos el resize banda a banda
    for i in range(21):
        band = ds3.GetRasterBand(i+1).ReadAsArray()
        res_data = resize(band, (round(ds3.RasterYSize/factor), round(ds3.RasterXSize/factor)))
        outdata.GetRasterBand(i+1).WriteArray(res_data)
    ds = None
    ds3 = None
     
    print('Resized file has been created from file {}'.format(folder_crop_s3.split('/')[-1]))
    
    
    
    
def down_cropping(tile_name, data_path, output_path, obj_s3_shape = None):
    
    """
    Crop of files, by considering that the Sentinel-2 one should have a width-height number of pixels multiple of 900,
    and the Sentinel-3 one should have a width-height number of pixels multiple of 30
    ----------
    Parameters
    tile_name: str
        Name of the region of interest
    data_path: str
        Path of the input data
    output_path
        Path of the data after being cropped: str
    """

    region_path = os.path.join(data_path, tile_name)
    files = os.listdir(region_path)
    tile_path_s2 = os.path.join(region_path, next((file for file in files if file.startswith("S2")), None))
    tile_path_s3 = os.path.join(region_path, next((file for file in files if file.startswith("S3")), None))

    output_region_path = os.path.join(output_path, tile_name)
    if not os.path.isdir(output_region_path):
        os.mkdir(output_region_path)
    tile_output_path_s2 = os.path.join(output_region_path, next((file for file in files if file.startswith("S2")), None))
    tile_output_path_s3 = os.path.join(output_region_path, next((file for file in files if file.startswith("S3")), None))
    
    # obtengo la imagen de sentinel-2
    ds_s2 = gdal.Open(tile_path_s2)
    width_s2 = ds_s2.RasterXSize
    height_s2 = ds_s2.RasterYSize
    gt_s2 = ds_s2.GetGeoTransform()
    ds_s2 = None
    if obj_s3_shape:
        fact1 = obj_s3_shape[1]
        fact2 = obj_s3_shape[0]
    else:
        fact1 = int(width_s2/900)
        fact2 = int(height_s2/900)   
        
    # calculo las nuevas coordenadas, pidiendo que sean múltiplo de 900 (el 1e-5 es para evitar problemas de redondeo)
    coord_s2 = {'W': gt_s2[0], 'N': gt_s2[3], 'E': gt_s2[0] + (1e-5+fact1*900)*gt_s2[1], 'S': gt_s2[3] + (1e-5+fact2*900)*gt_s2[5]} 

    # recortamos la imagen y la añadimos a la nueva carpeta
    with rasterio.open(tile_path_s2) as src_s2:
        window_s2 = from_bounds(coord_s2['W'], coord_s2['S'], coord_s2['E'], coord_s2['N'], src_s2.transform)
        kwargs_s2 = src_s2.meta.copy()
        kwargs_s2.update({
            'height': window_s2.height,
            'width': window_s2.width,
            'transform': rasterio.windows.transform(window_s2, src_s2.transform)})

        with rasterio.open(tile_output_path_s2, 'w', **kwargs_s2) as dst_s2:
            dst_s2.write(src_s2.read(window=window_s2))
     
    src_s2.closed
    dst_s2.closed
    
    # obtengo la imagen de sentinel-3
    ds_s3 = gdal.Open(tile_path_s3)
    width_s3 = ds_s3.RasterXSize
    height_s3 = ds_s3.RasterYSize
    gt_s3 = ds_s3.GetGeoTransform()
    ds_s3 = None

    # calculo las nuevas coordenadas, pidiendo que sean múltiplo de 30
    coord_s3 = {'W': gt_s3[0], 'N': gt_s3[3], 'E': gt_s3[0] + (1e-5+fact1*30)*gt_s3[1], 'S': gt_s3[3] + (1e-5+fact2*30)*gt_s3[5]} 

    # recortamos la imagen y la añadimos a la nueva carpeta
    with rasterio.open(tile_path_s3) as src_s3:
        window_s3 = from_bounds(coord_s3['W'], coord_s3['S'], coord_s3['E'], coord_s3['N'], src_s3.transform)
        kwargs_s3 = src_s3.meta.copy()
        kwargs_s3.update({
            'height': window_s3.height,
            'width': window_s3.width,
            'transform': rasterio.windows.transform(window_s3, src_s3.transform)})

        with rasterio.open(tile_output_path_s3, 'w', **kwargs_s3) as dst_s3:
            dst_s3.write(src_s3.read(window=window_s3))
            
    src_s3.closed
    dst_s3.closed
    
    print('{} files have been cropped'.format(tile_name))
    


def resampling(tile_name, data_path, output_path, upscale_factor): 
    
    """
    Resize image file in a factor of upscale_factor
    ----------
    Parameters
    tile_name: str
        Name of the region of interest
    data_path: str
        Path of the input image
    output_path: str
        Path of the resized image
    upscale_factor: float
        Factor of the resize
    """

    region_path = os.path.join(data_path, tile_name)   
    files = os.listdir(region_path)
    tile_path_s2 = os.path.join(region_path, next((file for file in files if file.startswith("S2")), None))
    tile_path_s3 = os.path.join(region_path, next((file for file in files if file.startswith("S3")), None))

    output_region_path = os.path.join(output_path, tile_name)
    if not os.path.isdir(output_region_path):
        os.mkdir(output_region_path)
    tile_output_path_s2 = os.path.join(output_region_path, next((file for file in files if file.startswith("S2")), None))
    tile_output_path_s3 = os.path.join(output_region_path, next((file for file in files if file.startswith("S3")), None))
    
    with rasterio.open(tile_path_s2) as dataset:
        
        # resample data to target shape using upscale_factor
        data = dataset.read(
            out_shape=(dataset.count, int(dataset.height * upscale_factor), int(dataset.width * upscale_factor)),
            resampling=Resampling.bilinear 
        )
        
        print('Shape before resample:', dataset.shape)
        print('Shape after resample:', data.shape[1:])
        
        # scale image transform
        dst_transform = dataset.transform * dataset.transform.scale((dataset.width / data.shape[-1]), (dataset.height / data.shape[-2]))

         # set properties for output
        dst_kwargs = dataset.meta.copy()
        dst_kwargs.update(
            {"crs": dataset.crs, "transform": dst_transform, "width": data.shape[-1], "height": data.shape[-2], "nodata": -9999}
        )
        
        # Write outputs
        with rasterio.open(tile_output_path_s2, "w", **dst_kwargs) as dst:
            # iterate through bands
            for i in range(data.shape[0]):
                dst.write(data[i], i+1)

    with rasterio.open(tile_path_s3) as dataset:
        
        # resample data to target shape using upscale_factor
        data = dataset.read(
            out_shape=(dataset.count, int(dataset.height * upscale_factor), int(dataset.width * upscale_factor)),
            resampling=Resampling.bilinear 
        )
        
        print('Shape before resample:', dataset.shape)
        print('Shape after resample:', data.shape[1:])
        
        # scale image transform
        dst_transform = dataset.transform * dataset.transform.scale((dataset.width / data.shape[-1]), (dataset.height / data.shape[-2]))

         # set properties for output
        dst_kwargs = dataset.meta.copy()
        dst_kwargs.update(
            {"crs": dataset.crs, "transform": dst_transform, "width": data.shape[-1], "height": data.shape[-2], "nodata": -9999}
        )
        
        # Write outputs
        with rasterio.open(tile_output_path_s3, "w", **dst_kwargs) as dst:
            # iterate through bands
            for i in range(data.shape[0]):
                dst.write(data[i], i+1)

    print('{} files have been downsampled'.format(tile_name))
                
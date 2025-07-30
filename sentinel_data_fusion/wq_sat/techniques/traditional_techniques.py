"""
Author: Daniel García Díaz & María Peña Fernández
Email: garciad@ifca.unican.es & penam@ifca.unican.es
Institute of Physics of Cantabria (IFCA)
Advanced Computing and e-Science
Date: Sep 2018 - June 2023
"""


import cv2
import numpy as np
import wq_sat.utils.gdal_utils as gdal_utils
from osgeo import gdal
import pywt
import os

def bicubic_interpolation(tile_name, data_path, output_path, output_name):
    """
    tile_path_s2: str
        Path of the Sentinel-2 image
    tile_path_s3: str
        Path of the Sentinel-3 image
    output_path: str
        Path where upsampled image is saved
    """

    region_path = os.path.join(data_path, tile_name)
    files = os.listdir(region_path)
    tile_path_s2 = os.path.join(region_path, next((file for file in files if file.startswith("S2")), None))
    tile_path_s3 = os.path.join(region_path, next((file for file in files if file.startswith("S3")), None))    
    
    # obtenemos la proyección y las transformadas de la imagen de sentinel-2
    ds_s2 = gdal.Open(tile_path_s2)
    coord_geotransform = ds_s2.GetGeoTransform()
    coord_geoprojection = ds_s2.GetProjection()
    ds_s2 = None

    data_bands = {}
    
    # guardamos en un array la imagen de Sentinel-3
    data_bands[300] = []
    ds_s3 = gdal.Open(tile_path_s3)
    data_bands[300] = ds_s3.ReadAsArray()
    data_bands[300] = np.moveaxis(data_bands[300], source=0, destination=2)
    ds_s3 = None  

    list_bands = []
    sr_bands = {}
    w,h = data_bands[300][:,:,0].shape[:2]

    # hacemos interpolación bicúbica banda a banda
    for i in range(21):
        band = cv2.resize(data_bands[300][:,:,i], (h*30, w*30) , interpolation=cv2.INTER_CUBIC)
        list_bands.extend([band])

    sr_bands[300] = list_bands
    
    # guardamos los nombres de las bandas SR y su descripción en diccionarios
    res_to_bands = {300: ['Oa01','Oa02','Oa03','Oa04','Oa05','Oa06','Oa07','Oa08','Oa09',
                          'Oa10','Oa11','Oa12','Oa13','Oa14','Oa15','Oa16','Oa17','Oa18',
                          'Oa19','Oa20','Oa21']}
    band_desc = {'Oa01': 'Oa01', 'Oa02': 'Oa02', 'Oa03': 'Oa03', 'Oa04': 'Oa04',
                 'Oa05': 'Oa05', 'Oa06': 'Oa06', 'Oa07': 'Oa07', 'Oa08': 'Oa08',
                 'Oa09': 'Oa09', 'Oa10': 'Oa10', 'Oa11': 'Oa11', 'Oa12': 'Oa12',
                 'Oa13': 'Oa13', 'Oa14': 'Oa14', 'Oa15': 'Oa15', 'Oa16': 'Oa16',
                 'Oa17': 'Oa17', 'Oa18': 'Oa18', 'Oa19': 'Oa19', 'Oa20': 'Oa20',
                 'Oa21': 'Oa21'}

    # Join the non-empty super resolved bands
    sr, validated_sr_bands = [], []
    for k, v in sr_bands.items():
        sr.append(v)
        validated_sr_bands += res_to_bands[k]
    sr = np.concatenate(sr, axis=2)

    # Create the lists of output variables to save
    output_bands, output_desc, output_shortnames = [], [], []

    for bi, bn in enumerate(validated_sr_bands):
        output_bands.append(sr[bi, :, :])
        output_desc.append("SR" + band_desc[bn])
        output_shortnames.append("SR" + bn)

    # guardamos la imagen SR
    gdal_utils.save_gdal_test(output_path=os.path.join(output_path, output_name),
                             bands=output_bands,
                             descriptions = output_desc,
                             geotransform=coord_geotransform,
                             geoprojection=coord_geoprojection,
                             file_format='GTiff')
    
    print('Bicubic interpolation on file from {} has been done \n'.format(tile_name))

    
    
    
def trwh_ihs(Rband, Gband, Bband, P):
    """
    Rband: arr
    Gband: arr
    Bband: arr
    P: arr
        Pansharpening band
    """
    w, h = P.shape

    #matriz de transformacion
    m = np.array([[1/3, 1/3, 1/3],
                  [-np.sqrt(2)/6, -np.sqrt(2)/6, (2*np.sqrt(2))/6], 
                  [1/np.sqrt(2), -1/np.sqrt(2), 0]]).astype('float32')
    
    #Transformar la imagen RGB en componentes IHS
    I = (m[0][0] * Rband) + (m[0][1] * Gband) + (m[0][2] * Bband)
    H = (m[1][0] * Rband) + (m[1][1] * Gband) + (m[1][2] * Bband)
    S = (m[2][0] * Rband) + (m[2][1] * Gband) + (m[2][2] * Bband)
    
    #Igualar histogramas: misma media y desviacion estandar
    a = np.nanstd(I) / np.nanstd(P)
    b = np.nanmean(I) - (np.nanstd(I) / np.nanstd(P)) * np.nanmean(P)
    Pe = a * P + b
    
    #Aplicar TRWH a la componente I hasta el segundo nivel descomposición
    coeffsI = pywt.wavedec2(I, 'db4', level=2)
    #Aplicar TRWH a la componente P hasta el segundo nivel descomposición
    coeffsPe = pywt.wavedec2(Pe, 'db4', level=2)
    
    #Nueva matriz concatenando los coeficientes cA2i, cV2p, cH2p ycD2p, cV1p, cH1p y cD1p. 
    new_coeffs = coeffsPe
    new_coeffs[0] = coeffsI[0]
    #transformada inversa de la TRHW
    IP = pywt.waverec2(new_coeffs, 'db4')
    IP = cv2.resize(IP, (h, w) , interpolation=cv2.INTER_CUBIC)
    
    #Matriz de transformacion
    t = np.array([[1, -1/np.sqrt(2), 1/np.sqrt(2)],
                  [1, -1/np.sqrt(2), -1/np.sqrt(2)],
                  [1, np.sqrt(2), 0]]).astype('float32')

    #Reconstruccion de las bandas con la nueva matriz IP
    Rf = (t[0][0] * IP) + (t[0][1] * H) + (t[0][2] * S)    
    Gf = (t[1][0] * IP) + (t[1][1] * H) + (t[1][2] * S)
    Bf = (t[2][0] * IP) + (t[2][1] * H) + (t[2][2] * S)
    
    return [Rf, Gf, Bf]




def pansharpening(tile_name, data_path, output_path, output_name):
    """
    tile_path_s2: str
        Path of the Sentinel-2 image
    tile_path_s3: str
        Path of the Sentinel-3 image
    output_path: str
        Path where upsampled image is saved
    """    

    region_path = os.path.join(data_path, tile_name)
    files = os.listdir(region_path)
    tile_path_s2 = os.path.join(region_path, next((file for file in files if file.startswith("S2")), None))
    tile_path_s3 = os.path.join(region_path, next((file for file in files if file.startswith("S3")), None))   
    data_bands = {}
    
    # guardamos los datos de sentinel-2
    ds_s2 = gdal.Open(tile_path_s2)
    coord_geotransform = ds_s2.GetGeoTransform()
    coord_geoprojection = ds_s2.GetProjection()
    data_bands[10] = ds_s2.ReadAsArray()
    data_bands[10] = np.moveaxis(data_bands[10], source=0, destination=2)
    P1 = ds_s2.GetRasterBand(1).ReadAsArray() 
    P2 = ds_s2.GetRasterBand(2).ReadAsArray()
    P3 = ds_s2.GetRasterBand(3).ReadAsArray()
    
    # hacemos la media de las tres RGB como banda de pansharpening
    P = (P1+P2+P3)/3
    ds_s2 = None

    # guardamos la imagen de Sentinel-3 en un array
    data_bands[300] = []
    ds_s3 = gdal.Open(tile_path_s3)
    data_bands[300] = ds_s3.ReadAsArray()
    data_bands[300] = np.moveaxis(data_bands[300], source=0, destination=2)
    ds_s3 = None  

    w, h = P.shape
    sr_bands = {}
    list_bands = []

    # aumentamos el tamaño de las bandas de Sentinel-3 para que tengan el de P
    for i in range(0,19,3):
        band1 = cv2.resize(data_bands[300][:,:,i], (h, w) , interpolation=cv2.INTER_CUBIC)
        band2 = cv2.resize(data_bands[300][:,:,i+1], (h, w) , interpolation=cv2.INTER_CUBIC)
        band3 = cv2.resize(data_bands[300][:,:,i+2], (h, w) , interpolation=cv2.INTER_CUBIC)
        list_bands.extend(trwh_ihs(band1, band2, band3, P))
        
    sr_bands[300] = list_bands
    
    # guardamos los nombres de las bandas SR y su descripción en diccionarios
    res_to_bands = {300: ['Oa01','Oa02','Oa03','Oa04','Oa05','Oa06','Oa07','Oa08','Oa09',
                          'Oa10','Oa11','Oa12','Oa13','Oa14','Oa15','Oa16','Oa17','Oa18',
                          'Oa19','Oa20','Oa21']}
    band_desc = {'Oa01': 'Oa01', 'Oa02': 'Oa02', 'Oa03': 'Oa03', 'Oa04': 'Oa04',
                 'Oa05': 'Oa05', 'Oa06': 'Oa06', 'Oa07': 'Oa07', 'Oa08': 'Oa08',
                 'Oa09': 'Oa09', 'Oa10': 'Oa10', 'Oa11': 'Oa11', 'Oa12': 'Oa12',
                 'Oa13': 'Oa13', 'Oa14': 'Oa14', 'Oa15': 'Oa15', 'Oa16': 'Oa16',
                 'Oa17': 'Oa17', 'Oa18': 'Oa18', 'Oa19': 'Oa19', 'Oa20': 'Oa20',
                 'Oa21': 'Oa21'}

    # Join the non-empty super resolved bands
    sr, validated_sr_bands = [], []
    for k, v in sr_bands.items():
        sr.append(v)
        validated_sr_bands += res_to_bands[k]
    sr = np.concatenate(sr, axis=2)

    # Create the lists of output variables to save
    output_bands, output_desc, output_shortnames = [], [], []

    for bi, bn in enumerate(validated_sr_bands):
        output_bands.append(sr[bi, :, :])
        output_desc.append("SR" + band_desc[bn])
        output_shortnames.append("SR" + bn)

    # guardamos la imagen SR
    gdal_utils.save_gdal_test(output_path=os.path.join(output_path, output_name),
                             bands=output_bands,
                             descriptions = output_desc,
                             geotransform=coord_geotransform,
                             geoprojection=coord_geoprojection,
                             file_format='GTiff')
    
    print('Pansharpening on file from {} has been done \n'.format(tile_name))
   
    
    
    
"""
This is a file to manipulate the patches needed to train/test models

Date: February 2019 & June 2023
Author: Ignacio Heredia & María Peña
Email: iheredia@ifca.unican.es & penam@ifca.unican.es
"""

from __future__ import division
from random import randrange
from math import ceil
import os
import numpy as np
import skimage.measure
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter
from tqdm import tqdm
from osgeo import gdal
import shutil


def recompose_images(a, border, size=None):
    """
    Recompose an image from the patches
    a: arr
        Contains all the patches once they have been upsampled
    border: int
        Number of pixels around the patch         
    """
    if a.shape[0] == 1:
        images = a[0]
    else:
        # This is done because we do not mirror the data at the image border
        patch_size = a.shape[2] - border*2
        x_tiles = int(ceil(size[1]/float(patch_size)))
        y_tiles = int(ceil(size[0]/float(patch_size)))
        
        # Initialize image
        images = np.zeros((size[0], size[1], a.shape[3])).astype(np.float32)
        current_patch = 0
        for y in range(0, y_tiles):
            ypoint = y * patch_size
            if ypoint > size[0] - patch_size:
                ypoint = size[0] - patch_size
            for x in range(0, x_tiles):
                xpoint = x * patch_size
                if xpoint > size[1] - patch_size:
                    xpoint = size[1] - patch_size 
                if current_patch < a.shape[0]:
                    images[ypoint:ypoint+patch_size, xpoint:xpoint+patch_size, :] = a[current_patch, border:a.shape[1]-border, border:a.shape[2]-border, :]
                current_patch += 1
    
    return images



def interp_patches(lr_image, hr_image_shape):
    """
    Resize of the lr patch received to a certain shape
    lr_image: arr
        Patch to be resized
    hr_image_shape: tuple
        Shape of the resized patch
    """
    interp = np.zeros((lr_image.shape[0:2] + hr_image_shape[2:4])).astype(np.float32)
    for k in range(lr_image.shape[0]):
        for w in range(lr_image.shape[1]):
            interp[k, w] = resize(image=lr_image[k, w],
                                  output_shape=hr_image_shape[2:4],
                                  mode='reflect')  # bilinear
    return interp



def get_test_patches(data_bands, patch_size=128, border=4, interp=True):
    """
    Compute patches when testing the model
    data_bands: arr
        Array to take patches from
    patch_size: int
        Size of the patches created
    border: int
        Number of pixels around the patch
    interp: bool
        Whether to resize patches or not
    """
    
    resolutions = data_bands.keys()
    max_res, min_res = max(resolutions), min(resolutions)
    scales = {res: int(res/min_res) for res in resolutions}  # scale with respect to minimum resolution  e.g. {10: 1, 300:30}
    inv_scales = {res: int(max_res/res) for res in resolutions}  # scale with respect to maximum resolution e.g. {10: 30, 300:1}
    
    # Adapt the borders and patchsizes for each scale
    patch_size = int(patch_size/scales[max_res]) * scales[max_res]  # make patchsize compatible with all scales
    borders = {res: border//scales[res] for res in resolutions}
    patch_sizes = {res: (patch_size//scales[res], patch_size//scales[res]) for res in resolutions}
    
    # Mirror the data at the borders to have the same dimensions as the input
    padded_bands = {}
    for res in resolutions:
        tmp_border = border // scales[res]
        padded_bands[res] = np.pad(data_bands[res], ((tmp_border, tmp_border), (tmp_border, tmp_border), (0, 0)), mode='symmetric')
       
    # Compute the number of patches
    P_i = padded_bands[max_res].shape[0] - 2 * borders[max_res]
    P_j = padded_bands[max_res].shape[1] - 2 * borders[max_res]
    Q_i = patch_sizes[max_res][0] - 2 * borders[max_res]
    Q_j = patch_sizes[max_res][1] - 2 * borders[max_res]
    
    patchesAlongi = P_i // Q_i
    patchesAlongj = P_j // Q_j
    nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)
       
    range_i = np.arange(0, patchesAlongi) * Q_i
    range_j = np.arange(0, patchesAlongj) * Q_j
    if not np.mod(P_i, Q_i) == 0:
        range_i = np.append(range_i, padded_bands[max_res].shape[0] - patch_sizes[max_res][0])
    if not np.mod(P_j, Q_j) == 0:
        range_j = np.append(range_j, padded_bands[max_res].shape[1] - patch_sizes[max_res][0])
            
    # Save the patches
    images = {res: np.zeros((nr_patches, padded_bands[res].shape[2]) + patch_sizes[res]).astype(np.float32) for res in resolutions}   
       
    pCount = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):
            upper_left_i = ii
            upper_left_j = jj
            crop_point = [upper_left_i,
                          upper_left_j,
                          upper_left_i + patch_sizes[max_res][0],
                          upper_left_j + patch_sizes[max_res][1]]
            for res in resolutions:
                tmp_crop_point = [p*inv_scales[res] for p in crop_point]
                tmp_cr_image = padded_bands[res][tmp_crop_point[0]:tmp_crop_point[2], tmp_crop_point[1]:tmp_crop_point[3]]
                to_image = np.moveaxis(tmp_cr_image, source=2, destination=0)
                if to_image.shape == images[res][pCount].shape:
                    images[res][pCount] = np.moveaxis(tmp_cr_image, source=2, destination=0)  # move to channels first
            pCount += 1
           
    if interp:
        for res in resolutions:
            images[res] = interp_patches(images[res], images[min_res].shape)
    
    return images



def downPixelAggr(img, SCALE=30):
    """
    Downsample image
    img: arr
        Array to be downsampled
    SCALE: arr
        Scale of the size reduction
    """
    if len(img.shape) == 2:
        img = np.expand_dims(img, axis=-1)
    img_blur = np.zeros(img.shape)

    # Filter the image with a Gaussian filter
    for i in range(0, img.shape[2]):
        img_blur[:, :, i] = gaussian_filter(img[:, :, i], 1/SCALE)
    
    # New image dims
    new_dims = tuple(s//SCALE for s in img.shape)
    img_lr = np.zeros(new_dims[0:2]+(img.shape[-1],))

    # Iterate through all the image channels with avg pooling (pixel aggregation)
    for i in range(0, img.shape[2]):
        img_lr[:, :, i] = skimage.measure.block_reduce(img_blur[:, :, i], (SCALE, SCALE), np.mean)

    return img_lr



def upsample_patches(lr_patch, output_shape):
    """
    Make the bilinear upsampling before feeding the network
    lr_patch: shape(H, W, bands)
    output_shape: shape(H, W, bands)
    """
    up_patches = np.zeros((*output_shape,lr_patch.shape[2])).astype(np.float32)
    for k in range(lr_patch.shape[2]):
        up_patches[:,:,k] = resize(image=lr_patch[:,:,k],
                                   output_shape=output_shape,
                                   mode='reflect')
    return up_patches



def save_random_patches(gt, lr, save_path, num_patches=None):
    """

    Parameters
    ----------
    gt : numpy array
        Array with the ground truth data of the maximal resolution
    lr : dict
        Containing low resolution bands
    save_path : str
        Path where patches are saved
    num_patches : int
        Number of patches to take from the image. If None the number will be proportional to the image size.
    """
    resolutions = lr.keys()
    max_res, min_res = max(resolutions), min(resolutions)
    scales = {res: int(res/min_res) for res in resolutions}  # scale with respect to minimum resolution  e.g. {10: 1, 20: 2, 60: 6}
    inv_scales = {res: int(max_res/res) for res in resolutions}  # scale with respect to maximum resolution e.g. {10: 6, 20: 3, 60: 1} or {10: 2, 20: 1}

    PATCH_SIZE_LR = (4, 4)
    i = 0

    if num_patches == None:

        for upper_left_x in range(lr[max_res].shape[0]- PATCH_SIZE_LR[0]+1):
            for upper_left_y in range(lr[max_res].shape[1] - PATCH_SIZE_LR[1]+1):
                crop_point_lr = [upper_left_x,
                                 upper_left_y,
                                 upper_left_x + PATCH_SIZE_LR[0],
                                 upper_left_y + PATCH_SIZE_LR[1]]
    
                # Create patches for HR map (label)
                mult_factor = scales[max_res]
                crop_point = [p * mult_factor for p in crop_point_lr]
                tmp_label = gt[crop_point[0]:crop_point[2], crop_point[1]:crop_point[3]]
    
                # Create patches for LR maps
                tmp_images = {}
                for res in lr.keys():
                    mult_factor = inv_scales[res]
                    crop_point = [p * mult_factor for p in crop_point_lr]
                    tmp_images[res] = lr[res][crop_point[0]:crop_point[2], crop_point[1]:crop_point[3]]
    
                # Make the bilinear upsampling
                for res in lr.keys():
                    tmp_images[res] = upsample_patches(tmp_images[res], output_shape=tmp_images[min_res].shape[:2])
    
                # Save patches to numpy binaries
                for res in lr.keys():
                    np.save(os.path.join(save_path, 'input{}_{}'.format(res, i)),
                            tmp_images[res])
    
                np.save(os.path.join(save_path, 'label{}_{}'.format(max_res, i)),
                        tmp_label)
                i += 1
    else:

            for upper_left_x in range(lr[max_res].shape[0]- PATCH_SIZE_LR[0]+1):
                for upper_left_y in range(lr[max_res].shape[1] - PATCH_SIZE_LR[1]+1):
                    if num_patches <= (lr[max_res].shape[0]- PATCH_SIZE_LR[0]+1)*(lr[max_res].shape[1] - PATCH_SIZE_LR[1]+1):
                        num_patches = num_patches
                    else:
                        num_patches = (lr[max_res].shape[0]- PATCH_SIZE_LR[0]+1)*(lr[max_res].shape[1] - PATCH_SIZE_LR[1]+1)
                        
                    while i < num_patches:
                        
                        crop_point_lr = [upper_left_x,
                                         upper_left_y,
                                         upper_left_x + PATCH_SIZE_LR[0],
                                         upper_left_y + PATCH_SIZE_LR[1]]
            
                        # Create patches for HR map (label)
                        mult_factor = scales[max_res]
                        crop_point = [p * mult_factor for p in crop_point_lr]
                        tmp_label = gt[crop_point[0]:crop_point[2], crop_point[1]:crop_point[3]]
            
                        # Create patches for LR maps
                        tmp_images = {}
                        for res in lr.keys():
                            mult_factor = inv_scales[res]
                            crop_point = [p * mult_factor for p in crop_point_lr]
                            tmp_images[res] = lr[res][crop_point[0]:crop_point[2], crop_point[1]:crop_point[3]]
            
                        # Make the bilinear upsampling
                        for res in lr.keys():
                            tmp_images[res] = upsample_patches(tmp_images[res], output_shape=tmp_images[min_res].shape[:2])
            
                        # Save patches to numpy binaries
                        for res in lr.keys():
                            np.save(os.path.join(save_path, 'input{}_{}'.format(res, i)),
                                    tmp_images[res])
            
                        np.save(os.path.join(save_path, 'label{}_{}'.format(max_res, i)),
                                tmp_label)
                        i += 1
            
    print('{} patches have been created'.format(i))
        
        
def create_patches(tiles, max_res, data_dir=None, save_dir=None, roi_x_y=None, num_patches=None): ######## HAGO CAMBIOS AQUÍ
    """

    Parameters
    ----------
    tiles : list
        Files to take patches of
    max_res : int
        Maximum resolution considered
    roi_x_y : list
        x and y positions (coordinates)
    tiles_dir : str
        Directory of the files
    save_dir : str
        Where to save the patches
    num_patches : int
        Number of patches to be created

    Returns
    -------
    """
    if isinstance(tiles, str):
        tiles = [tiles]

    for tile in tiles:

        region_path = os.path.join(data_dir, tile)
        files = os.listdir(region_path)
        
        # Encontrar los archivos S2 y S3
        s2_file = next((file for file in files if file.startswith("S2")), None)
        s3_file = next((file for file in files if file.startswith("S3")), None)

        # Obtener las rutas completas de los archivos S2 y S3
        s2_path = os.path.join(region_path, s2_file)
        s3_path = os.path.join(region_path, s3_file)
        
        # Clearing previous patches (if any)
        output_dir = os.path.join(save_dir, tile)
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        os.mkdir(output_dir)
        
        data_bands = {}
        ds_s2 = gdal.Open(s2_path)
        data_bands[10] = ds_s2.ReadAsArray()
        data_bands[10] = np.moveaxis(data_bands[10], source=0, destination=2)
        ds_s2 = None
        
        data_bands[300] = []
        ds_s3 = gdal.Open(s3_path)
        data_bands[300] = ds_s3.ReadAsArray()
        data_bands[300] = np.moveaxis(data_bands[300], source=0, destination=2)
        ds_s3 = None

        # Normalize pixel values and put image in float32 format
        for res in data_bands.keys():
            data_bands[res] = data_bands[res].astype(np.float32)
            for num_band in range(data_bands[res].shape[-1]):
                data_bands[res][:,:,num_band] = (data_bands[res][:,:,num_band] - np.mean(data_bands[res][:,:,num_band])) / np.sqrt(np.var(data_bands[res][:,:,num_band]))
                
        # Define the scales of the problem
        resolutions = data_bands.keys()
        max_res, min_res = max(resolutions), min(resolutions)
        scales = {res: int(res / min_res) for res in resolutions}  # scale with respect to minimum resolution  e.g. {10: 1, 300:30}
        inv_scales = {res: int(max_res / res) for res in resolutions}  # scale with respect to maximum resolution e.g. {10: 30, 300: 1} 
        scale = scales[max_res]

        # Check if image has fill_value pixel
        tmp_band = data_bands[min_res][:, :, 0]
        if np.sum(tmp_band == -9999) > 0:
            print('The selected image has some [fill_value] pixels')

        # Crop GT maps so that they can be correctly downscaled
        old_H, old_W = data_bands[max_res].shape[:2]  # size of the smallest map
        new_H, new_W = int(old_H/scale) * scale, int(old_W/scale) * scale
        for res, bands in data_bands.items():
            tmp_H, tmp_W = inv_scales[res] * new_H, inv_scales[res] * new_W
            data_bands[res] = bands[:tmp_H, :tmp_W, :]

        # Create the LR maps
        gt = data_bands
        lr = {res: None for res in gt.keys()}
        for res, gt_maps in gt.items():
            lr[res] = downPixelAggr(gt_maps, SCALE=scale)

        save_random_patches(gt=gt[max_res], lr=lr, save_path=output_dir, num_patches=num_patches)       



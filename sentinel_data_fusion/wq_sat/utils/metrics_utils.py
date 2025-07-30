import numpy as np

def mae(ms, f):
    """
    Mean Absolute Error
    ms: arr
        Original band
    f: arr
        Super´-resolved band
    """
    
    x = ms.flatten()
    y = f.flatten()
    
    n = len(x)
    dif_abs = abs(y - x)
    mae = np.sum(dif_abs) / n

    return round(mae, 4)


def rmse(ms, f):
    """
    Root Mean Squared Error
    ms: arr
        Original band
    f: arr
        Super´-resolved band
    """
    
    x = ms.flatten()
    y = f.flatten()
    
    n = len(x)
    dif_sq = (y - x)**2
    rmse = np.sqrt(np.sum(dif_sq) / n)

    return round(rmse, 4)

def sam(org_img: np.ndarray, pred_img: np.ndarray, convert_to_degree: bool = True) -> float:
    """
    Spectral Angle Mapper which defines the spectral similarity between two spectra
    """
        
    numerator = np.sum(np.multiply(pred_img, org_img), axis=0)
    denominator = np.linalg.norm(org_img, axis=0) * np.linalg.norm(pred_img, axis=0)
    val = np.clip(numerator / denominator, -1, 1)
    sam_angles = np.arccos(val)
    if convert_to_degree:
        sam_angles = np.rad2deg(sam_angles)

    return round(np.nan_to_num(np.mean(sam_angles)),2)


def sre(org_img: np.ndarray, pred_img: np.ndarray):
    """
    Signal to Reconstruction Error Ratio
    """

    org_img = org_img.astype(np.float32)
   
    # if image is a gray image - add empty 3rd dimension for the .shape[2] to exist
    if org_img.ndim == 2:
        org_img = np.expand_dims(org_img, axis=0)
        pred_img = np.expand_dims(pred_img, axis=0)
        
    sre_final = []
    for i in range(org_img.shape[0]):
        numerator = np.square(np.mean(org_img[i, :, :]))
        denominator = (np.linalg.norm(org_img[i, :, :] - pred_img[i, :, :])) / (
            org_img.shape[1] * org_img.shape[2]
        )
        sre_final.append(numerator / denominator)

    return round(10 * np.log10(np.mean(sre_final)),2)
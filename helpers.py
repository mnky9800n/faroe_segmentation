import numpy as np

def remove_data_based_on_radius(data, mask_value=-1):
    """
    creates a mask based on the radius of the sample to remove
    outlying data in the scan which is meaningless and not part
    of the sample
    """
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    
    cx = data.shape[0] // 2
    cy = data.shape[1] // 2
    r = data.shape[0] // 2
    
    mask = (x[np.newaxis,:] - cx)**2 + (y[:, np.newaxis] - cy)**2 > r**2
    
    data_maskradius = data.astype(np.float64)
    data_maskradius[mask] = mask_value
    
    return data_maskradius

def zscore(data):
    
    data_flat = data.flatten()
    data_flat = data_flat[~np.isnan(data_flat)]
    
    mean = np.mean(data_flat)
    std = np.std(data_flat)
    
    return (data - mean)/std
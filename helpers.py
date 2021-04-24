import numpy as np
from scipy.signal import find_peaks
import cv2

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

def get_peaks_means_stds(img, height, width):
    """
    finds peaks in histogram across all values in image
    """
    h, e = np.histogram(img[~np.isnan(img)].flatten(), bins=1000)
#     peaks, other = find_peaks(h, height=height, width=width)
    peaks, other = find_peaks(h, height=height, width=width, distance=50)
    means = e[peaks]
    stds = np.std(img[~np.isnan(img)].flatten())/3
    
    return dict(zip(('hist', 'edges', 'peaks', 'means', 'stds'), (h, e, peaks, means, stds)))

def label_by_peaks(img, peaks):
    """
    labels image pixels by nearest peak location
    """
    zeros = np.zeros_like(img)
    std = peaks['stds']
    for n, m in enumerate(peaks['means']):
        plower = m - std
        pupper = m + std
        x, y = np.where(np.logical_and(img>plower, img<pupper))
        zeros[x, y] = n + 1
        
#     return helpers.remove_data_based_on_radius(zeros, mask_value=np.nan)
    return remove_data_based_on_radius(zeros, mask_value=np.nan)

def fix_brightness(img):
    """
    Fixes shadowing issues
    
    https://stackoverflow.com/questions/44047819/increase-image-brightness-without-overflow/44054699#44054699
    """
    
    dilated_img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated_img, 21)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = diff_img.copy()
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    return norm_img
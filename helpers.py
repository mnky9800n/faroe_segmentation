import numpy as np
from scipy.signal import find_peaks
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks
from scipy.signal import argrelextrema
from skimage import exposure
from scipy import ndimage


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

def label_pipeline(img):
    """
    Applies functions for labeling image
    """
    # fix brightness and shadowing issues
    # https://stackoverflow.com/questions/44047819/increase-image-brightness-without-overflow/44054699#44054699

#     norm_img = helpers.fix_brightness(img)
    norm_img = fix_brightness(img)
    
    # apply histogram equalization for vesicle picking
    # https://scikit-image.org/docs/dev/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py
    imarr_hist = exposure.equalize_hist(norm_img)
    
    # remove data from outside of core
    # imarr_maskradius = helpers.remove_data_based_on_radius(imarr, mask_value=np.nan)
#     imarr_hist = helpers.remove_data_based_on_radius(imarr_hist, mask_value=np.nan)
    imarr_hist = remove_data_based_on_radius(imarr_hist, mask_value=np.nan)

    sigma = 1
    blurs = gaussian_filter(imarr_hist, sigma=sigma)

    # calculate peaks
#     peaks_etc = helpers.get_peaks_means_stds(blurs, height=1e3, width=4)
    peaks_etc = get_peaks_means_stds(blurs, height=1e3, width=4)

    # label peaks
#     peak_labled = helpers.label_by_peaks(blurs, peaks_etc)
    peak_labled = label_by_peaks(blurs, peaks_etc)

    
    # TODO : function this out
    # find local minima in histogram to separate vesicles from noise
    z = np.copy(peak_labled)
    z[z == 1] = 0
    # z[z == 2] = 3
    # z[z == 3] = 0

    z = gaussian_filter(z, sigma=4)

    h, e = np.histogram(z.flatten(), bins=np.linspace(0, 2, 201))
    try:
        splitter = e[argrelextrema(h, np.less, order=10)][0]
    except:
#         splitter = e[argrelextrema(h, np.greater, order=10)][0]
        # set splitter at middle of distribution if we cannot
        # smartly find a local minimum to split on 
#         splitter = 1
        splitter = 1.25
    
    # label vesicles
    z[z >= splitter] = 2
    z[z < splitter] = 0

    # calculate global mean and rescale values

#     zim = helpers.zscore(helpers.remove_data_based_on_radius(img, np.nan))
#     zim = zscore(helpers.remove_data_based_on_radius(img, np.nan))
    zim = zscore(remove_data_based_on_radius(img, np.nan))

    # filter out all data that is not fracture

    zim_remove = zim.copy()
    zim_remove[zim_remove > -1.75] = 0
    # zim_remove[zim_remove > -4] = 0

    x1, y1 = zim_remove.nonzero()

    zeros = np.zeros_like(zim_remove)

    # label fractures
    zeros[x1, y1] = 1

    # replace all data as either 1 or 0 for fracture identification

    # footprint condition says value should be greater
    # in all directions
    footprint1 = np.array(
        [[1, 1, 1]
        ,[1, 0, 1]
        ,[1, 1, 1]]
    )

    footprint2 = np.array(
        [[1, 1, 1, 1, 1]
        ,[1, 1, 1, 1, 1]
        ,[1, 1, 0, 1, 1]
        ,[1, 1, 1, 1, 1]
        ,[1, 1, 1, 1, 1]]
    )

    # creates a window based on the given footprint
    # to compare neighbors and replace values on 
    # nearest maximum value
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.maximum_filter.html
    newim = ndimage.median_filter(zeros, footprint=footprint1, mode='constant')
#     newim = ndimage.median_filter(zeros, footprint=footprint2, mode='constant')
    newim = ndimage.maximum_filter(newim, footprint=footprint2, mode='constant')

#     imlabeled = helpers.remove_data_based_on_radius(newim, mask_value=np.nan)
    imlabeled = remove_data_based_on_radius(newim, mask_value=np.nan)

    # create new labeled image
    zeros = np.zeros_like(imlabeled)

    # label vesicles
    ves_x, ves_y = z.nonzero()
    zeros[ves_x, ves_y] = 1

    # label fractures
    frac_x, frac_y = imlabeled.nonzero()
    zeros[frac_x, frac_y] = 2

#     zeros = helpers.remove_data_based_on_radius(zeros, np.nan)
    zeros = remove_data_based_on_radius(zeros, np.nan)
    
    del frac_x, frac_y, ves_x, ves_y, imlabeled, newim, x1, y1, zim_remove, splitter, z, peak_labled, peaks_etc, blurs, imarr_hist, norm_img
    
    return zeros
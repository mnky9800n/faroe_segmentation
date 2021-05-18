import cv2
import numpy as np
import matplotlib.pyplot as plt
import helpers
from multiprocessing import Pool
from skimage import exposure
from datetime import datetime
import glob
from skimage.measure import label, regionprops_table, regionprops
import pandas as pd
from scipy import ndimage


def apply_otsu_dilate_erode(img):
    ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.dilate(img, np.ones((7, 7), np.uint8))
    img = cv2.erode(img, np.ones((7, 7), np.uint8))
    return img

def remove_convex_area_erode(img):
    labeled = label(helpers.remove_data_based_on_radius(abs(img-255)))
#     labeled = label(remove_data_based_on_radius(abs(img-255)))
    df = regionprops_table(labeled, properties=['convex_area', 'label'])
    df = pd.DataFrame(df)
    img = np.where(np.isin(labeled, df[df.convex_area > 10000].label.values), labeled, np.nan)
    img = cv2.erode(img, np.ones((7, 7), np.uint8))
    return img

def convert_to_binary(img):
    ones_i = np.where(np.isnan(img), 0, 1)
    return ones_i

def label_fractures(img):
    img_remove = helpers.remove_data_based_on_radius(img, mask_value=np.nan)
#     img_remove = remove_data_based_on_radius(img, mask_value=np.nan)
    zim = helpers.zscore(img_remove)
#     zim = zscore(img_remove)
    zim_remove = zim.copy()
    zim_remove[zim_remove > -2] = 0
    
    x1, y1 = zim_remove.nonzero()
    zeros = np.zeros_like(zim_remove)
    zeros[x1, y1] = 1
    return zeros

def median_max(img):
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
    img = ndimage.median_filter(img, footprint=footprint1, mode='constant')
    img = ndimage.maximum_filter(img, footprint=footprint2, mode='constant')
    return img

def remove_convex_area_eccentricity(img, convex_area=60, eccentricity=0.8):
    """
    Removes fractures based on convex area and eccentricity minimum values
    
    convex_area : Number of pixels of convex hull image, which is the smallest convex polygon that encloses the region.

    eccentricity : Eccentricity of the ellipse that has the same second-moments as the region. The eccentricity is the ratio of the focal distance (distance between focal points) over the major axis length. The value is in the interval [0, 1). When it is 0, the ellipse becomes a circle.
    """
    labeled_fracs = label(img)
    df = pd.DataFrame(regionprops_table(labeled_fracs, properties=[ 'convex_area', 'eccentricity', 'label']))
#     img = np.where(np.isin(labeled_fracs, df[(df.convex_area > 60) & (df.eccentricity > 0.8)].label.values), img, np.nan)
    img = np.where(np.isin(labeled_fracs, df[(df.convex_area > convex_area) & (df.eccentricity > eccentricity)].label.values), img, np.nan)
    return img

# def combine_vesicles_fractures(img):
def combine_vesicles_fractures(img, i_ves, i_frac):
    """
    combines vesicle and fracture images labeling 
    vesicles as 1 and fractures as 2 and all other
    part of the rock as 0.
    """
    img_zeros = np.zeros_like(img)

    xves, yves = np.nonzero(i_ves)

    img_zeros[xves, yves] = 1

    xfrac, yfrac = np.nonzero(np.nan_to_num(i_frac))

    img_zeros[xfrac, yfrac] = 2
    
    img_zeros = helpers.remove_data_based_on_radius(img_zeros, -1)
    
    return img_zeros


def multi_figures(file):
    img = cv2.imread(file,0)
    i_ves = apply_otsu_dilate_erode(img)
    i_ves = remove_convex_area_erode(i_ves)
    i_ves = convert_to_binary(i_ves)
    
    i_frac = label_fractures(img)
    i_frac = median_max(i_frac)
#     i_frac = remove_convex_area_eccentricity(i_frac)
    i_frac = remove_convex_area_eccentricity(i_frac, convex_area=60, eccentricity=0.2)
    
#     fig, ax = helpers.plot_image(img, cmap='Greys')
    fig, ax = helpers.plot_image(img, cmap='Greys', vmin=80, vmax=100)
    ax.imshow(i_ves, alpha=0.5, cmap='PRGn', interpolation='none')
    ax.imshow(i_frac, alpha=0.75, cmap='plasma_r', interpolation='none')
    fig.savefig('/media/sda/data/labeled/F20_10b/'+file[19:]+'_labeled.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    data = combine_vesicles_fractures(img, i_ves, i_frac)
    np.save(arr=data, file='/media/sda/data/labeled/F20_10b/'+file[19:]+'_labeled.npy')
    del data, i_frac, i_ves, img 
    print(file, 'processing complete')
    
if __name__=='__main__':
    
    filenames = [f for f in glob.glob('benoitdata/F20_10_b/*')]

    pool = Pool(maxtasksperchild=10)

    start = datetime.now()
    print('started at ', start)

    pool.map(multi_figures, filenames)
    pool.close()

    print('this took', datetime.now()-start)
    pool.close()

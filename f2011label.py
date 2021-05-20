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

def multi_figures(file):
    img = cv2.imread(file, 0)
    zim = helpers.zscore(helpers.remove_data_based_on_radius(img, np.nan))
    labeled = label_fractures(zim)
    
    labeled = labeled.astype(np.float32)
    labeled[labeled==0] = np.nan
    labeled = np.nan_to_num(labeled, nan=0)
    labeled = helpers.remove_data_based_on_radius(labeled, -1)

    fig, ax = helpers.plot_image(img, cmap='Greys')
    ax.imshow(labeled, cmap='Purples_r', alpha=0.5)
    fig.savefig('/media/sda/data/labeled/F20_11/'+file[18:]+'_labeled.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    np.save(arr=labeled, file='/media/sda/data/labeled/F20_11/'+file[18:]+'_labeled.npy')
    del img, zim, labeled
    print(file, 'processing complete')


if __name__=='__main__':
    
    filenames = [f for f in glob.glob('benoitdata/F20-11/*')]

    pool = Pool(maxtasksperchild=10)

    start = datetime.now()
    print('started at ', start)

    pool.map(multi_figures, filenames)
    pool.close()

    print('this took', datetime.now()-start)
    pool.close()
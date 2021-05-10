import cv2
import numpy as np
import matplotlib.pyplot as plt
import helpers
from multiprocessing import Pool
from skimage import exposure
from datetime import datetime
import glob

# filenames = !ls benoitdata/F20-14A/

filenames = [f for f in glob.glob('benoitdata/F20-14A/*')]

def multi_figures(file):
    # img = cv2.imread('benoitdata/F20-14A/'+file,0)
    img = cv2.imread(file,0)
#     i = label_pipeline(img)
    i = helpers.label_pipeline(img)
#     np.save(arr=i, file='labeled/F20_14/'+file+'_labeled.npy')
    np.save(arr=i, file='/media/sda/data/labeled/F20_14/'+file[19:]+'_labeled.npy')
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].imshow(exposure.equalize_hist(img), cmap='plasma')
    ax[1].imshow(i, cmap='plasma')
    fig.tight_layout()
    fig.savefig('/media/sda/data/labeled/F20_14/'+file[19:]+'_labeled.png', dpi=300, bbox_inches='tight')
    plt.close()
    del img, i, fig, ax
    
pool = Pool(maxtasksperchild=10)

start = datetime.now()
print('started at ', start)

pool.map(multi_figures, filenames)
pool.close()

print('this took', datetime.now()-start)
pool.close()

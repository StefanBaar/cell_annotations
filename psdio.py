import datetime
from tqdm import tqdm
from glob import glob

import numpy as np
import cv2

from psd_tools import PSDImage

from joblib import Parallel, delayed

import warnings
warnings.filterwarnings("ignore")

def mCPU(func, var, n_jobs=20,verbose=10):
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(func)(i) for i in var)

def layer_to_RGB(layer):
    return (layer[:, :, :-1].astype(int)//41*41).astype("uint8")

def get_colors(arr):
    return np.unique(arr.reshape(-1, 3), axis=0)[:-1]

def large_psd_to_npy(path,CPUS=16):
    psd    = PSDImage.open(path)
    lnr    = len(psd)

    if lnr>=CPUS:
        n = CPUS
    else:
        n = lnr

    def process(i):
        return np.asarray(i.composite(psd.viewbox))

    return np.stack(mCPU(process,psd,n,0))


def psd_layesrs_to_npy(path):
    psd   = PSDImage.open(path)
    LAYERS= []

    for n,i in enumerate(tqdm(psd)):
        #print(n,i)
        image = np.asarray(i.composite(psd.viewbox))
        LAYERS.append(image)

    return np.asarray(LAYERS)

def get_masks_from_layer(psd, alpha_th=1):
    """get image, and binary masks from psd layers
       first layer:      image
       second layer:     contaminants
       remaining layers: cells (one cell per image)"""

    def get_binary_mask(layers, alpha_th):
        mask = np.zeros_like(layers[:,:,:,0])
        mask[layers[:,:,:,3]>alpha_th] = 1
        return mask

    image   = psd[0]
    blayers = get_binary_mask(psd[1:], alpha_th)

    return image, blayers

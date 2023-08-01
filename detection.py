import numpy as np
import cv2

from astropy.stats import SigmaClip
from photutils.background import Background2D, MedianBackground

from scipy import ndimage
from skimage.filters import sobel
from skimage.segmentation import watershed
from skimage import morphology



def norm(data):
    return data.astype(float)/data.max()

def detect_cells(image,bkg_sigma=4.0, edge_sigma=3, hsize=400, obsize=400):
    #### bgk removal
    sigma_clip    = SigmaClip(sigma=bkg_sigma)
    bkg_estimator = MedianBackground()
    bkg           = Background2D(image,
                                 (40, 40),
                                 filter_size=(3, 3),
                                 sigma_clip=sigma_clip,
                                 bkg_estimator=bkg_estimator).background

    bkg_image     = image-bkg+0.5

    ##### sobel filter
    edges         = sobel(bkg_image)

    ##### watershed segmentation
    markers       = np.zeros_like(edges,dtype="uint8")
    foreground, background = 1, 2

    markers[edges < np.std(edges)/edge_sigma] = background
    markers[edges > np.std(edges)/edge_sigma] = foreground

    ws   = 2.-watershed(edges, markers)


    mask = morphology.remove_small_holes(ws.astype(bool),
                                         area_threshold=hsize)

    mask = morphology.remove_small_objects(mask,
                                           min_size=obsize)

    return mask

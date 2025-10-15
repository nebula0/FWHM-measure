import numpy as np
def kappaSigmaClipping(kappa, iter, img_ori):
    '''
    do Kappa-Sigma Clipping, returm the mask in 2D array,
    True is pixel belong to background
    False is pixel belong to source
    img_ori: original image in 2D array
    kappa: kappa value, kappa * std will classify as source
    iter: iteration times
    '''
    img = img_ori
    mask = img > np.min(img) # create mask
    keep_pixel = img
    for i in range(iter):
        mean = np.mean(keep_pixel)
        std = np.std(keep_pixel)
        
        mask = img < mean + kappa*std
        keep_pixel = img[mask]
    bg_mean = np.mean(img[mask])
    bg_std = np.std(img[mask])
    return mask, bg_mean, bg_std
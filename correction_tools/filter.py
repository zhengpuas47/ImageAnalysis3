from scipy.ndimage.filters import gaussian_filter
import numpy as np

def gaussian_deconvolution(im, gfilt_size=2, niter=1):
    """Gaussian blurred image divided by image itself"""

    decon_im = im.copy().astype(np.float32)
    for _iter in np.arange(niter):
        decon_im = decon_im / gaussian_filter(decon_im, gfilt_size)
    
    return decon_im


def gaussian_high_pass_filter(image, sigma=5, truncate=2):
    """Apply gaussian high pass filter to given image"""
    lowpass = gaussian_filter(image, sigma, mode='nearest', truncate=truncate)
    gauss_highpass = image - lowpass
    gauss_highpass[lowpass > image] = 0
    return gauss_highpass
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats 
import numpy as np
import matplotlib.pyplot as plt
from astropy.visualization import simple_norm

def find_source(data, sigma=5.0, fwhm=7.0, threshold_times=5.):
    '''
    Detects sources in the given 2D data array using DAOStarFinder.
    Parameters:
    - data: 2D array of the image data.
    - sigma:  mean ± sigma times std will be viewed as outliers.
    - fwhm: Full width at half maximum for the Gaussian kernel. You need to guess it based on your image.
    - threshold_times: std * threshold_times will be used as the detection threshold.
    Returns:
    - sources: Table of detected sources with their properties.
    '''
    mean, median, std = sigma_clipped_stats(data, sigma=sigma) 
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold_times*std)  
    sources = daofind(data - median)  
    for col in sources.colnames:  
        if col not in ('id', 'npix'):
            sources[col].info.format = '%.2f'  # for consistent table output
    return sources

from photutils.aperture import CircularAperture

def plot_source(data, sources, radius = 10, percent=99.5, **kwargs):
    '''
    Plots the data with detected sources marked by circular apertures.
    Parameters:
    - data: 2D array of the image data.
    - sources: Table of detected sources with 'xcentroid' and 'ycentroid
    - radius: Radius of the circular apertures.
    - percent: Percentile for normalization.
    - kwargs: Additional keyword arguments for plt.subplots().
    Returns:
    return fig, ax
    '''
    positions = np.transpose((sources['xcentroid'], sources['ycentroid']))
    apertures = CircularAperture(positions, r=radius)
    norm = simple_norm(data, stretch='sqrt', percent=percent)

    fig, ax = plt.subplots(**kwargs)  # 使用 subplots 來獲取圖形和軸對象
    img = ax.imshow(data, cmap='viridis', origin='lower', norm=norm, interpolation='nearest')
    apertures.plot(color='white', lw=2, alpha=0.9, ax=ax)
    fig.colorbar(img, ax=ax)  # 添加色條

    # 返回圖形和軸對象
    return fig, ax
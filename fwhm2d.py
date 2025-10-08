from astropy.modeling import models, fitting
import numpy as np
import matplotlib.pyplot as plt
from photutils.aperture import EllipticalAperture
from astropy.visualization import simple_norm

def FWHM2d(Qtable, data, sub_size_half=6):
    '''
    use Gaussian2D to fit the source and get FWHM in x and y direction
    Parameters:
    - Qtable: Table of detected sources with 'xcentroid' and 'ycentroid
    - data: 2D array of the image data.
    - sub_size_half: Half size of the square region around each source to 
    fit the Gaussian, e.g., 6 means 12x12 pixels. you need to guess it based on your image.
    Returns:
    - fwhm_x_list: List of FWHM in x direction for each source
    - fwhm_y_list: List of FWHM in y direction for each source
    - mask: Boolean list indicating whether the FWHM was successfully calculated for each source (True) 
    or skipped due to being too close to the image edge (False).


    '''
    fwhm_x_list = []
    fwhm_y_list = []
    mask = []
    count = 1
    for i in range(len(Qtable)):
        xc, yc = Qtable['xcentroid'][i], Qtable['ycentroid'][i]  

        # select area around the source
        x_min, x_max = int(xc - sub_size_half), int(xc + sub_size_half)
        y_min, y_max = int(yc - sub_size_half), int(yc + sub_size_half)
        # If the box is out of image, skip this source
        if x_min < 0 or x_max < 0 or y_min < 0 or y_max < 0:
            fwhm_x_list.append(-1)
            fwhm_y_list.append(-1)
            mask.append(False)
            continue
        mask.append(True)
        sub_data = data[y_min:y_max, x_min:x_max]
        yp, xp = sub_data.shape


        # Generate grid of same size like box to put the fit on
        y, x, = np.mgrid[:yp, :xp]
        # Declare what function you want to fit to your data
        apmlitude = sub_data.max() - sub_data.min()
        x_mean = xc - x_min
        y_mean = yc - y_min

        f_init = models.Gaussian2D(amplitude=apmlitude, x_mean=x_mean, y_mean=y_mean)
        # Declare what fitting function you want to use
        fit_f = fitting.LevMarLSQFitter()

        # Fit the model to your data (box)
        fitted_g = fit_f(f_init, x, y, sub_data, maxiter=10000)
        fit_info = fit_f.fit_info
        if count == 1:
            print(fit_info.keys())
            print(fit_info['message'])
            print(fit_info['ierr'])
            count -= 1

        # 计算 FWHM
        fwhm_x = 2.355 * fitted_g.x_stddev.value
        fwhm_y = 2.355 * fitted_g.y_stddev.value

        fwhm_x_list.append(fwhm_x)
        fwhm_y_list.append(fwhm_y)
        # plot_2d_gaussian_profile(sub_data, fitted_g)

    return fwhm_x_list, fwhm_y_list, mask


def plot_source_fwhm_ellips(data, sources, **kwargs):
    '''
    Plots the data with detected sources marked by elliptical apertures based on FWHM.
    Parameters:
    - data: 2D array of the image data.
    - sources: Table of detected sources with 'xcentroid', 'ycentroid',
        'x_fwhm' and 'y_fwhm' columns.
    - kwargs: Additional keyword arguments for plt.subplots().
    Returns:
    return fig, ax
    '''
    apertures = []
    for source in sources:
        position = (source['xcentroid'], source['ycentroid'])
        a = source['x_fwhm']  # x軸半徑
        b = source['y_fwhm']  # y軸半徑
        theta = 0  # 橢圓角度（如有可用 source['theta']）
        aperture = EllipticalAperture(position, a=a, b=b, theta=theta)
        apertures.append(aperture)

    norm = simple_norm(data, stretch='sqrt', percent=99)
    fig, ax = plt.subplots(**kwargs)
    img = ax.imshow(data, cmap='viridis', origin='lower', norm=norm, interpolation='nearest')
    fig.colorbar(img, ax=ax)
    for aperture in apertures:
        aperture.plot(color='white', lw=1.5, alpha=0.9, ax=ax)
    return fig, ax

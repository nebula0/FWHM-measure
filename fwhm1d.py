import numpy as np
from typing import Literal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset

def fit_gaussian_xy(image, sources, sourceID, axis: Literal['x', 'y'], avg_pixel_num = 4):
    '''
    Fit a Gaussian profile to the x or y cut of a source in the image.
    Parameters:
    - image: 2D array of the image data.
    - sources: Table of detected sources with 'xcentroid', 'ycentroid',
        'x_fwhm' and 'y_fwhm' columns.
    - sourceID: Index of the source in the sources table to fit.
    - axis: 'x' or 'y' to indicate which cut to fit.
    - avg_pixel_num: Number of pixels to average in the perpendicular direction.
        
    Returns:
    - parm: Fitted parameters [amp, mu, sigma, offset].
    - main_rel_range: Relative pixel positions from the source center.
    - intensity_profile: Averaged intensity profile along the cut.
    - x0 or y0: Centroid position of the source in the cut direction.
    - y_half_range or x_half_range: Half range used for averaging in the perpendicular direction
    
    '''
    source = sources[sourceID]
    if axis == 'x':
        # x profile: x ± 2FWHMx, y ± avg_pixel_num//2
        x0 = source['xcentroid']
        y0 = source['ycentroid']
        x_fwhm = source['x_fwhm']
        y_fwhm = source['y_fwhm']
        y_half_range = avg_pixel_num//2
        y_min = int(max(0, y0 - y_half_range))
        y_max = int(min(image.shape[0], y0 + y_half_range + 1))
        x_min = int(max(0, x0 - x_fwhm))
        x_max = int(min(image.shape[1], x0 + x_fwhm + 1))
        cut_data = image[y_min:y_max, x_min:x_max]
        intensity_profile = np.mean(cut_data, axis=0)
        main_range = np.arange(x_min, x_max)
        main_rel_range = main_range - x0
        sigma_guess = x_fwhm / 2.355
    elif axis == 'y':
        # y profile: y ± 2FWHMy, x ± avg_pixel_num//2
        x0 = source['xcentroid']
        y0 = source['ycentroid']
        x_fwhm = source['x_fwhm']
        y_fwhm = source['y_fwhm']
        x_half_range = avg_pixel_num//2
        x_min = int(max(0, x0 - x_half_range))
        x_max = int(min(image.shape[1], x0 + x_half_range + 1))
        y_min = int(max(0, y0 - y_fwhm))
        y_max = int(min(image.shape[0], y0 + y_fwhm + 1))
        cut_data = image[y_min:y_max, x_min:x_max]
        intensity_profile = np.mean(cut_data, axis=1)
        main_range = np.arange(y_min, y_max)
        main_rel_range = main_range - y0
        sigma_guess = y_fwhm / 2.355
    else:
        raise ValueError("axis must be 'x' or 'y'")

    p0 = [np.max(intensity_profile)-np.min(intensity_profile), 0, sigma_guess, np.min(intensity_profile)]
    parm＿optimal, parm_covarience = curve_fit(gaussian, main_rel_range, intensity_profile, p0=p0, maxfev=10000)
    # calculate residule RMS
    residule = intensity_profile - gaussian(main_rel_range, *parm＿optimal)
    rms = np.sqrt(np.mean(residule**2))


    
    return parm＿optimal, main_rel_range, intensity_profile, (x0 if axis=='x' else y0), (y_half_range if axis=='x' else x_half_range), rms

def plot_xy_cut_gaussian(sources, sourceID, parm, x_rel, intensity_profile, x0, axis:Literal['x', 'y'], ax=None):
    '''
    plot the x cut +- 2 FWHMx, avg(+- factor * FWHMy) of the source and fit with gaussian
    '''
    assert axis in ('x', 'y'), "axis 只能是 'x' 或 'y'"
    if ax is None:
        figsize=(6,4)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure # 表示外面傳進來的 ax，要交給外面管理

    source = sources[sourceID]

    # 繪圖
    ax.plot(x_rel, intensity_profile, 'g.', label='Data')
    ax.plot(x_rel, gaussian(x_rel, *parm), 'k--', label='Gaussian fit')
    
    # 從擬合參數取出 mu, sigma
    amp, mu, sigma, offset = parm # mu 是中心, sigma 是標準差
    fwhm = 2.355 * sigma

    # 畫出中心線
    ax.axvline(mu, color='gray', linestyle='-', label='Fit center')

    # --- FWHM 標示區塊 ---
    amp, mu, sigma, offset = parm
    fwhm = 2.355 * sigma
    half_max = offset + amp / 2

    # 畫灰底矩形 (標示 FWHM 區域)
    ax.axvspan(mu - fwhm/2, mu + fwhm/2, 
               ymin=0,
               color='gray', alpha=0.2)
    # 雙箭頭顯示 FWHM
    ax.annotate(
        '', 
        xy=(mu + fwhm/2, half_max), 
        xytext=(mu - fwhm/2, half_max),
        arrowprops=dict(arrowstyle='<->', color='b', lw=1.5)
    )
    # 在箭頭下方加文字標籤
    ax.text(mu, half_max*0.7, f"FWHM = {fwhm:.2f}", 
            color='b', ha='center', va='bottom', fontsize=10)




    # 標示 Y 標準差
    # ax.errorbar(x_rel, intensity_profile, yerr=intensity_std, fmt='none',ecolor='g', label='Data ± 1σ', alpha=0.3)

    ax.set_xlabel('X distance from center [pixels]')
    ax.set_ylabel('Intensity')
    ax.legend()
    
    return fig, ax

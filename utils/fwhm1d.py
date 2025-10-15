import numpy as np
from typing import Literal
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def gaussian(x, amp, mu, sigma, offset):
    return amp * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset


from scipy.optimize import curve_fit
import numpy as np

def fit_gaussian_xy_new(image, sources, sourceID, axis: Literal['x', 'y'], avg_pixel_num=4):
    '''
    Fit a Gaussian profile to the x or y cut of a source in the image.
    '''
    source = sources[sourceID]
    x0 = source['xcentroid']
    y0 = source['ycentroid']
    x_fwhm = source['x_fwhm']
    y_fwhm = source['y_fwhm']

    if axis == 'x':
        y_half_range = avg_pixel_num // 2
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
        x_half_range = avg_pixel_num // 2
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

    # ---- 🔹 防呆處理：檢查資料是否足夠擬合 ----
    if len(main_rel_range) < 5 or np.allclose(intensity_profile, intensity_profile[0]):
        return [np.nan, np.nan, np.nan, np.nan], main_rel_range, intensity_profile, (x0 if axis=='x' else y0), \
               (y_half_range if axis=='x' else x_half_range), np.nan

    # ---- 初始參數猜測 ----
    p0 = [np.max(intensity_profile)-np.min(intensity_profile), 0, sigma_guess, np.min(intensity_profile)]

    try:
        parm_optimal, parm_covariance = curve_fit(
            gaussian, main_rel_range, intensity_profile, p0=p0, maxfev=20000
        )
        residule = intensity_profile - gaussian(main_rel_range, *parm_optimal)
        rms = np.sqrt(np.mean(residule**2))

        # ---- 🔹 檢查結果是否為 nan ----
        if np.any(np.isnan(parm_optimal)) or np.any(np.isinf(parm_optimal)):
            raise RuntimeError("Invalid fit result (nan/inf)")

    except Exception as e:
        # ---- 🔹 出錯時仍回傳安全值 ----
        parm_optimal = [np.nan, np.nan, np.nan, np.nan]
        rms = np.nan

    return parm_optimal, main_rel_range, intensity_profile, (x0 if axis=='x' else y0), \
           (y_half_range if axis=='x' else x_half_range), rms


def plot_fit_gaussian_and_data(parm, x_rel, intensity_profile, pixel_len_um, ax=None):
    '''
    plot the x cut +- 2 FWHMx, avg(+- factor * FWHMy) of the source and fit with gaussian
    '''
    if ax is None:
        figsize=(6,4)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure # 表示外面傳進來的 ax，要交給外面管理


    # 繪圖
    ax.plot(x_rel, intensity_profile, 'g.', label='Data')
    ax.plot(x_rel, gaussian(x_rel, *parm), 'k--', label='Gaussian fit')
    
    # 從擬合參數取出 mu, sigma
    amp, mu, sigma, offset = parm # mu 是中心, sigma 是標準差
    fwhm = 2.355 * sigma

    # 畫出中心線
    ax.axvline(mu, color='gray', linestyle='-', label='Fit center')

    # --- FWHM 標示區塊 ---
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
    ax.text(mu, half_max - 0.1 * amp, f"FWHM = {fwhm*pixel_len_um:.2f} um", 
            color='b', ha='center', va='bottom', fontsize=10)

    ax.legend()
    
    return fig, ax

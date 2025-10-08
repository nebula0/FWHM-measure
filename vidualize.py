# norm with sqrt and plot center percent
from astropy.visualization import simple_norm
import matplotlib.pyplot as plt

def plot_picture(data, percent=99.5, **kwargs):
    '''
    plot the fits file data
    img:
    get data by 
    hdu = fits.open('SLT_EPIC210651981.fits')
    data = hdu[0].data

    percent:
    only show middle {percent} percent data
    '''
    norm = simple_norm(data, stretch='sqrt', percent=percent)
    fig, ax = plt.subplots(**kwargs)  # 使用 subplots 來獲取圖形和軸對象
    img = ax.imshow(data, cmap='viridis', origin='lower', norm=norm, interpolation='nearest')
    fig.colorbar(img, ax=ax)  # 添加色條
    return fig, ax

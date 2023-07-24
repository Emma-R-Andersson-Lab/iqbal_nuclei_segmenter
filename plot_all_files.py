import pathlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import numpy as np

from nuclei_segmenter import loader


DATA_DIR = pathlib.Path(r'C:\Users\agucor\Karolinska Institutet\Afshan Iqbal - EdU staining IGF1 Exp')
IMG_DIR = DATA_DIR / 'data'
RES_DIR = DATA_DIR / 'results/figures/all_organoids'

def load_img(filepath):
    file = loader.get_file(filepath)
    return loader.get_image(file)


def load_and_plot(filepath, ax):
    img = load_img(filepath)
    scale = loader.get_pixel_size(loader.get_file(filepath))

    img = np.max(img, axis=0)

    this_im = ax.imshow(img, rasterized=True)
    ax.set_axis_off()
    ax.set_title(filepath.stem)

    scalebar = AnchoredSizeBar(ax.transData,
                               100/(scale['X'] * 10 ** 6), '', 'lower left', 
                               pad=0.1,
                               color='white',
                               frameon=False,
                               size_vertical=10)

    ax.add_artist(scalebar)
    plt.colorbar(this_im, ax=ax)


for folder_path in IMG_DIR.iterdir():
    print(f'Analyzing {folder_path}')
    file_list = list(folder_path.glob('*.czi'))
    
    subplot_width = np.floor(np.sqrt(len(file_list))).astype(int)
    subplot_height = np.ceil(np.sqrt(len(file_list))).astype(int)
    
    fig, axs = plt.subplots(subplot_height, subplot_width, figsize=(4*subplot_width, 3*subplot_height))
    axs = axs.flatten()
    
    for file, ax in zip(file_list, axs):
        load_and_plot(file, ax)
    
    plt.subplots_adjust()
    
    save_dir = RES_DIR / f'{folder_path.stem}.png'
    print(f'Saving at {save_dir}')
    plt.savefig(save_dir, dpi=300)
    plt.close()
    
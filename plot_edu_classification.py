from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import pandas as pd
from itertools import chain

from nuclei_segmenter.loader import get_nuclei_size_df, get_file, get_image, get_edu_image, get_labeled
from nuclei_segmenter.quantify import find_edu_nuclei_in_df
from nuclei_segmenter.vis import get_contours, labeled_colormap, my_positive_colormap, my_negative_colormap, relabel

DATA_DIR = pathlib.Path(r'C:\Users\agucor\Karolinska Institutet\Afshan Iqbal - EdU staining IGF1 Exp')
BIG_DATA_DIR = pathlib.Path(r'C:\Users\agucor\Karolinska Institutet\Afshan Iqbal - EdU staining IGF1 Exp\sparse nuclei')

IMG_DIR = DATA_DIR / 'data'
SEG_DIR = DATA_DIR / 'segmented'
BIG_IMG_DIR = BIG_DATA_DIR / 'data'
BIG_SEG_DIR = BIG_DATA_DIR / 'segmented'
PDF_DIR = DATA_DIR / 'results/figures/edu_classification'


nuclei_size_original = get_nuclei_size_df(DATA_DIR)
nuclei_size_big = get_nuclei_size_df(BIG_DATA_DIR)
nuclei_size = pd.concat([nuclei_size_original, nuclei_size_big], ignore_index=True)
nuclei_size = find_edu_nuclei_in_df(nuclei_size, intensity_col='intensity_mean_edu')

def get_images(filepath):
    with get_file(filepath) as file:
        img = get_image(file)
        edu = get_edu_image(file)

    prop_df = nuclei_size.query(f"sample_name == '{filepath.stem}'")
    good_planes = prop_df.z.unique()
    if len(good_planes) == 0:
        return None, None
    return img[good_planes.astype(int)], edu[good_planes.astype(int)]


def plot_edu_classification(img, edu, edu_pos_labeled):
    fig, axs = plt.subplots(2, 2, figsize=(17, 17))

    axs[0, 0].imshow(np.max(img, axis=0))

    axs[0, 1].imshow(np.max(edu, axis=0))

    axs[0, 0].axis('off')
    axs[0, 1].axis('off')

    axs[1, 0].imshow(np.max(img, axis=0))

    contours = get_contours(np.max(labeled, axis=0))
    axs[1, 0].imshow(contours, cmap=labeled_colormap, vmin=0.1, interpolation='none')

    axs[1, 1].imshow(np.max(edu, axis=0))

    this_edu_class = np.max(edu_pos_labeled == 2, axis=0).astype(int)
    if (this_edu_class > 0).any():
        contours_positive = get_contours(this_edu_class)
        axs[1, 1].imshow(contours_positive, cmap=my_positive_colormap, vmin=0.1, interpolation='none')

    this_edu_class = np.max(edu_pos_labeled == 1, axis=0).astype(int)
    if (this_edu_class > 0).any():
        contours_negative = get_contours(this_edu_class)
        axs[1, 1].imshow(contours_negative, cmap=my_negative_colormap, vmin=0.1, interpolation='none')

    axs[1, 0].axis('off')
    axs[1, 1].axis('off')

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    return axs

for folder in chain(SEG_DIR.iterdir(), BIG_SEG_DIR.iterdir()):
    print(f'Preparing folder {folder}')
    if folder.is_relative_to(SEG_DIR):
        files = [filepath.relative_to(SEG_DIR) for filepath in folder.rglob('*.tiff')]
        prefix = ''
    elif folder.is_relative_to(BIG_SEG_DIR):
        files = [filepath.relative_to(BIG_SEG_DIR) for filepath in folder.rglob('*.tiff')]
        prefix = 'sparse_'
    else:
        raise ValueError

    with PdfPages(PDF_DIR / (prefix + folder.stem + '.pdf')) as pp:
        for file in files:
            print(f'Making page {file}')
            this_df = nuclei_size.query(f'sample_name == "{file.stem}"')
            
            img, edu = get_images(IMG_DIR / file.with_suffix('.czi'))
            if img is None and edu is None:
                continue

            if folder.is_relative_to(SEG_DIR):
                labeled = get_labeled(SEG_DIR / file)
            elif folder.is_relative_to(BIG_SEG_DIR):
                labeled = get_labeled(BIG_SEG_DIR / file)

            all_labels = np.arange(np.max(labeled))
            present_labels = [label if label in this_df.label.values else 0 for label in all_labels]
            edu_pos_labeled = relabel(labeled, zip(this_df.edu.values + 1, this_df.label.values))
            labeled = relabel(labeled, zip(present_labels, all_labels))

            plot_edu_classification(img, edu, edu_pos_labeled)
            plt.suptitle(file.stem)
            pp.savefig()
            plt.close()
from multiprocessing import Pool
import pathlib
from tqdm import tqdm

from nuclei_segmenter.loader import get_file, get_image, get_edu_image, get_labeled, get_pixel_size
from nuclei_segmenter.quantify import quantify


DATA_DIR = pathlib.Path(r'C:\Users\agucor\Karolinska Institutet\Afshan Iqbal - EdU staining IGF1 Exp')
IMG_DIR = DATA_DIR / 'data'
SEGM_DIR = DATA_DIR / 'segmented'
# DATA_DIR = pathlib.Path(r'C:\Users\Lab\OneDrive - Karolinska Institutet\Microwell plate experiments\EdU staining Non-IGF1 exp')
# SEGM_DIR = pathlib.Path(r'C:\Users\Lab\OneDrive - Karolinska Institutet\Microwell plate experiments\EdU staining Non-IGF1 exp\segmented')


file_list = [filepath for filepath in SEGM_DIR.rglob("*.tiff") if 'eleted' not in str(filepath)]


def requantify(segment_path):
    print('ReAnalyzing ' + segment_path.name)
    filepath = IMG_DIR / segment_path.parent.stem / (segment_path.stem + '.czi')

    file = get_file(filepath)

    print('Loading image')
    image = get_image(file)
    edu_image = get_edu_image(file)
    scale = get_pixel_size(file)
    labels = get_labeled(segment_path)

    print('Quantifying ' + segment_path.name)
    nuclei_props = quantify(image, labels, scale, extra_image=edu_image)

    prop_path = segment_path.with_name(filepath.stem + '.xlsx')
    print('Saving at ' + str(prop_path))
    nuclei_props.to_excel(prop_path)


if __name__ == '__main__':
    # with Pool(15) as p:
    #     for this in tqdm(p.imap_unordered(run_and_save, file_list)):
    #         pass

    for this in tqdm(map(requantify, file_list), total=len(file_list)):
        pass
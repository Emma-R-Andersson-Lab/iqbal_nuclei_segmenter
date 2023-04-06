from multiprocessing import Pool
import pathlib
from tqdm import tqdm

from nuclei_segmenter.loader import get_file, get_image, get_pixel_size
from nuclei_segmenter.segment import segment_nuclei, polish_segmentation
from nuclei_segmenter.quantify import quantify
from nuclei_segmenter.vis import save_img

#
# DATA_DIR = pathlib.Path(r'P:\c5_Andersson_Emma\Afshan\Nuclei Area quantication')
# SEGM_DIR = pathlib.Path(r'P:\c5_Andersson_Emma\Afshan\Nuclei Area quantication\segmented_nuclei\segmented')


DATA_DIR = pathlib.Path(r'C:\Users\agucor\Karolinska Institutet\Afshan Iqbal - EdU staining Non-IGF1 exp')
SEGM_DIR = pathlib.Path(r'C:\Users\agucor\Karolinska Institutet\Afshan Iqbal - EdU staining Non-IGF1 exp\segmented')
file_list = [filepath for filepath in DATA_DIR.rglob("*.czi") if 'Orthogonal' not in filepath.name]
# file_list = [DATA_DIR / 'Wt1223/WT1223 Peripheral-2.czi',
#              DATA_DIR / 'Wt1223/WT1223 Peripheral-8.czi']


def run_and_save(filepath):
    print('Analyzing ' + filepath.name)
    segment_path = SEGM_DIR / filepath.parent.stem / (filepath.stem + '.tiff')
    if segment_path.exists():
        print('Already segmented')
        return

    file = get_file(filepath)

    print('Loading image')
    image = get_image(file)
    scale = get_pixel_size(file)

    labels = segment_nuclei(image, scale)
    print('Saving at ' + str(segment_path))
    save_img(segment_path, labels)

    print('Quantifying ' + filepath.name)
    nuclei_props = quantify(image, labels, scale)

    prop_path = segment_path.with_name(filepath.stem + '.xlsx')
    print('Saving at ' + str(prop_path))
    nuclei_props.to_excel(prop_path)


if __name__ == '__main__':
    # with Pool(15) as p:
    #     for this in tqdm(p.imap_unordered(run_and_save, file_list)):
    #         pass

    for this in tqdm(map(run_and_save, file_list), total=len(file_list)):
        pass

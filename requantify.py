from multiprocessing import Pool
import pathlib
from tqdm import tqdm

from nuclei_segmenter.loader import get_file, get_image, get_labeled, get_pixel_size
from nuclei_segmenter.quantify import quantify


DATA_DIR = pathlib.Path(r'P:\c5_Andersson_Emma\Afshan\Nuclei Area quantication')
SEGM_DIR = pathlib.Path(r'P:\c5_Andersson_Emma\Afshan\Nuclei Area quantication\segmented_nuclei\segmented')
file_list = [filepath for filepath in SEGM_DIR.iterdir() if filepath.is_file()
             and filepath.suffix == '.tiff']


def requantify(segment_path):
    print('ReAnalyzing ' + segment_path.name)
    filepath = DATA_DIR / (segment_path.stem + '.czi')

    file = get_file(filepath)

    print('Loading image')
    image = get_image(file)
    scale = get_pixel_size(file)
    labels = get_labeled(segment_path)

    print('Quantifying ' + segment_path.name)
    nuclei_props = quantify(image, labels, scale)

    prop_path = segment_path.with_name(filepath.stem + '.xlsx')
    print('Saving at ' + str(prop_path))
    nuclei_props.to_excel(prop_path)


if __name__ == '__main__':
    # with Pool(15) as p:
    #     for this in tqdm(p.imap_unordered(run_and_save, file_list)):
    #         pass

    for this in tqdm(map(requantify, file_list), total=len(file_list)):
        pass
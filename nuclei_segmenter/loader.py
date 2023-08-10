from czifile import CziFile
import numpy as np
import pandas as pd
from tifffile import TiffFile


def get_file(filepath):
    return CziFile(filepath)


def get_pixel_size(file):
    metadata = file.metadata(raw=False)

    return {dim_dict['Id']: float(dim_dict['Value']) for dim_dict in
            metadata['ImageDocument']['Metadata']['Scaling']['Items']['Distance']}


def get_image(file):
    img = np.squeeze(file.asarray())
    if img.ndim > 3 or file.shape[file.axes.find('Z')] == 1:
        img = img[0]
    return img


def get_edu_image(file):
    img = np.squeeze(file.asarray())
    return img[1]


def get_labeled(label_filepath):
    return TiffFile(label_filepath).asarray()


def parse_condition(text):
    if 'wt' in text.lower():
        return 'WT'
    elif 'ndr' in text.lower():
        return 'Ndr'
    else:
        raise ValueError


def parse_cell_line(text):
    return text.split(' ')[0][-4:]


def parse_region(text):
    if 'peri' in text.lower():
        return 'Peripheral'
    elif 'hilar' in text.lower():
        return 'Hilar'
    else:
        raise ValueError


def parse_sample_number(text):
    return text.split('-')[-1]


def parse_igf1(text):
    if 'igf1' in text.lower():
        return 'IGF1'
    elif 'ctrl' in text.lower():
        return 'Control'
    else:
        raise ValueError


def get_nuclei_size_df(filepath):
    nuclei_size = pd.read_excel(filepath / 'nuclei_size.xlsx',
                                         sheet_name=None)
    dfs = []
    for sample_name, sample_df in nuclei_size.items():
        sample_df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'], inplace=True)
        sample_df['sample_name'] = sample_name
        dfs.append(sample_df)

    nuclei_size = pd.concat(dfs, ignore_index=True)

    nuclei_size['condition'] = nuclei_size.sample_name.apply(parse_condition)
    nuclei_size['cell_line'] = nuclei_size.sample_name.apply(parse_cell_line)
    nuclei_size['region'] = nuclei_size.sample_name.apply(parse_region)
    nuclei_size['sample_number'] = nuclei_size.sample_name.apply(parse_sample_number)

    nuclei_size = nuclei_size.astype({'label': int, 
                                      'condition': 'category', 
                                      'cell_line': 'category', 
                                      'region': 'category', 
                                      'sample_number': 'category', 
                                    })

    try:
        nuclei_size['igf1'] = nuclei_size.sample_name.apply(parse_igf1)
        nuclei_size = nuclei_size.astype({'igf1': 'category'})
    except ValueError:
        pass

    return nuclei_size
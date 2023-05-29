from czifile import CziFile
import numpy as np
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

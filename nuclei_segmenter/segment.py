from csbdeep.utils import normalize
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, morphology as morph, measure as meas
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from stardist.models import StarDist2D

from .loader import get_file, get_image, get_labeled, get_pixel_size


def my_kernel():
    kernel = np.zeros([3] + list(morph.disk(4).shape))
    small_size = morph.disk(2).shape[0]
    ini = small_size // 2
    kernel[0, ini:ini + small_size, ini:ini + small_size] = morph.disk(2)
    kernel[2, ini:ini + small_size, ini:ini + small_size] = morph.disk(2)
    kernel[1] = morph.disk(4)

    return kernel


def smooth_img(img):
    return filters.median(img, footprint=my_kernel())


def stardist_2D(img):
    if img.ndim > 2:
        labels = np.asarray([stardist_2D(this) for this in img])
    else:
        model = StarDist2D.from_pretrained('2D_versatile_fluo')
        labels, throw = model.predict_instances(normalize(img))

    return labels


def filter_by_intensity(img, labels, int_threshold=3000):
    filtered_labels = []
    for this_z, this_labels in zip(img, labels):
        props = pd.DataFrame(
            meas.regionprops_table(labels, intensity_image=img, properties=(['label', 'intensity_mean']))
        )

        this_filtered_labels = this_labels.copy()

        labels_to_delete = props.query('intensity_mean < %f' % int_threshold).label.values

        this_filtered_labels[np.logical_or.reduce([this_filtered_labels == to_del for to_del in labels_to_delete])] = 0
        filtered_labels.append(this_filtered_labels)

    return np.asarray(filtered_labels)


def label_iterator(labeled):
    for label in np.unique(labeled):
        if label == 0:
            continue
        yield label


def my_distance(labeled):
    distance = np.zeros_like(labeled, dtype=float)
    for label in label_iterator(labeled):
        distance += ndi.distance_transform_edt(labeled == label)
    return distance


def watershed_segmentation(labels, lat_scale=1/6):
    distance_between_nuclei = np.floor(2.5 / lat_scale).astype(int)
    if labels.ndim > 2:
        separated_labels = np.asarray([watershed_segmentation(this) for this in labels])
    else:
        distance = my_distance(labels)
        coords = peak_local_max(distance, footprint=morph.disk(distance_between_nuclei), labels=labels > 0,
                                min_distance=distance_between_nuclei)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers, _ = ndi.label(mask)
        separated_labels = watershed(-distance, markers, mask=labels)
    return separated_labels


def is_similar(labeled_1, labeled_2, threshold=0.9):
    return np.sum(np.logical_and(labeled_1.flatten(),
                                 labeled_2.flatten())) / np.sum(np.logical_or(labeled_1.flatten(),
                                                                              labeled_2.flatten())) > threshold


def connect_stacks(labels):
    new_labels = np.zeros_like(labels)

    new_labels[0] = labels[0].copy()
    label_count = np.max(new_labels)

    for z, new_labels_next in enumerate(labels[1:], 1):
        for label in label_iterator(new_labels_next):
            label_found = False

            similar_labels = []
            for old_label in label_iterator(new_labels[z - 1][new_labels_next == label]):
                if is_similar(new_labels_next == label, new_labels[z - 1] == old_label, threshold=0.6):
                    label_found = True
                    similar_labels.append(old_label)

            if not label_found:
                label_count += 1
                new_labels[z][new_labels_next == label] = label_count

            else:
                if len(similar_labels) > 1:
                    raise ValueError

                new_labels[z][new_labels_next == label] = similar_labels[0]

    return new_labels


def threshold_img(smooth_img):
    threshold = filters.threshold_li(smooth_img.flatten())
    return smooth_img > threshold


def clean_mask(mask, disk_size=5):
    polish_mask = np.asarray([morph.binary_opening(this_plane, footprint=morph.disk(disk_size))
                              for this_plane in mask])
    return polish_mask


def polish_segmentation(filepath, labeled_path):
    """Deprecated as segment_nuclei includes this"""
    file = get_file(filepath)

    print('Loading image')
    image = get_image(file)

    print('Smoothing image')
    smoothed = smooth_img(image)

    print('Thresholding smoothed image')
    mask = threshold_img(image)

    print('Cleaning mask')
    mask = clean_mask(mask)

    print('Applying mask')
    labeled = get_labeled(labeled_path)
    labeled = labeled * mask

    return labeled


def segment_nuclei_from_filepath(filepath):
    file = get_file(filepath)

    print('Loading image')
    image = get_image(file)
    scale = get_pixel_size(file)

    return segment_nuclei(image, scale)


def segment_nuclei(image, scale):
    # print('Smoothing image')
    # smoothed = smooth_img(image)

    print('Initial 2D segmentation')
    labels = stardist_2D(image)

    # print('Separating labels by shape')
    # labels = watershed_segmentation(labels, lat_scale=scale['X'])

    print('Connecting labels')
    labels = connect_stacks(labels)

    print('Thresholding smoothed image')
    mask = threshold_img(image)

    print('Cleaning mask')
    disk_size_pix = np.ceil(1/(scale['X'] * 10 ** 6)).astype(int)
    mask = clean_mask(mask, disk_size=disk_size_pix)

    print('Applying mask')
    labels = labels# * mask

    return labels

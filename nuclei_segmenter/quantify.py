import numpy as np
import pandas as pd
from skimage import measure as meas, morphology as morph, filters
from sklearn.mixture import GaussianMixture

from .loader import get_image, get_file, get_pixel_size, get_labeled


def get_2d_props(img, labeled):
    nuclei_props = []
    for this_z, (this_img, this_labeled) in enumerate(zip(img, labeled)):
        this_nuclei_props = meas.regionprops_table(this_labeled, intensity_image=this_img, properties=(
            'area', 'eccentricity', 'equivalent_diameter_area', 'intensity_mean', 'label'))
        this_nuclei_props = pd.DataFrame.from_dict(this_nuclei_props)
        this_nuclei_props['z'] = this_z
        nuclei_props.append(this_nuclei_props)

    return pd.concat(nuclei_props, ignore_index=True)


def get_3d_props(img, labeled):
    return pd.DataFrame.from_dict(meas.regionprops_table(labeled, intensity_image=img,
                                                         properties=('area', 'intensity_mean', 'label')))


def get_3d_extra_int(img, labeled):
    return pd.DataFrame.from_dict(meas.regionprops_table(labeled,
                                                         intensity_image=img,
                                                         properties=('intensity_mean', 'label'),
                                                         extra_properties=[intensity_median]))


def intensity_median(regionmask, intensity_image):
    return np.median(intensity_image[regionmask])


def get_all_props(img, labeled, extra_image=None):
    nuclei_props_2d = get_2d_props(img, labeled)
    nuclei_props_3d = get_3d_props(img, labeled)
    if extra_image is not None:
        nuclei_extra_int = get_3d_extra_int(extra_image, labeled)
        nuclei_props_3d = pd.merge(nuclei_props_3d, nuclei_extra_int,
                                   on='label', suffixes=('', '_edu'))
    nuclei_props_all = pd.merge(nuclei_props_2d, nuclei_props_3d,
                                on='label', suffixes=('_2D', '_3D'))

    return nuclei_props_all


def calculate_area(nuclei_props, scale):
    pixel_size = (scale['X'] * 10 ** 6) * (scale['Y'] * 10 ** 6)
    nuclei_props['area'] = nuclei_props.area_2D * pixel_size
    return nuclei_props


def calculate_volume(nuclei_props, scale):
    if 'Z' in scale:
        voxel_size = (scale['X'] * 10 ** 6) * (scale['Y'] * 10 ** 6) * (scale['Z'] * 10 ** 6)
        nuclei_props['volume'] = nuclei_props.area_3D * voxel_size
    return nuclei_props


def unify_labels_by_area(nuclei_props):
    inds = nuclei_props.groupby('label')['intensity_mean_2D'].idxmax()
    return nuclei_props.iloc[inds]


def is_good_plane(plane, solidity_threshold=0.6):
    organoid = morph.binary_dilation(plane > 0, footprint=morph.disk(20))
    try:
        return meas.regionprops(organoid.astype(int))[0]['solidity'] > solidity_threshold
    except IndexError:
        return False


def get_good_planes(labeled, threshold=0.6):
    return [z for z, plane in enumerate(labeled) if is_good_plane(plane, solidity_threshold=threshold)]


def get_gmm(vals, number_of_classes=3):
    model = GaussianMixture(n_components=number_of_classes)
    model.fit(vals.reshape(-1, 1))
    return model


def get_class_dict(model):
    means = model.means_.flatten()
    means_inds = np.argsort(means)
    if len(means) == 2:
        size = ['small', 'big']
        class_dict = {c: size[means_ind] for c, means_ind in enumerate(means_inds)}
    elif len(means) == 3:
        size = ['small', 'medium', 'big']
        class_dict = {c: size[means_ind] for c, means_ind in enumerate(means_inds)}
    else:
        raise ValueError('More than three classes used for classification')
    return class_dict


def get_separation(model, max_size=800):
    probas = model.predict_proba(np.arange(0, max_size, 1).reshape(-1, 1))
    class_dict = get_class_dict(model)
    inv_map = {v: k for k, v in class_dict.items()}
    size = ['small', 'medium', 'big'] if len(class_dict) == 3 else ['small', 'big']

    lims = []
    for i in range(1, len(class_dict)):
        small_name = size[i - 1]
        big_name = size[i]
        lim_name = small_name + ' to ' + big_name
        lim = np.argwhere(np.diff(np.sign(probas[:, inv_map[big_name]] - probas[:, inv_map[small_name]]))).flatten()
        lim = lim[~np.isin(lim, 0)]
        lims.append((lim_name, lim))

    return lims


def classify(model, vals):
    cs = model.predict(vals.reshape(-1, 1))
    class_dict = get_class_dict(model)
    return np.asarray([class_dict[c] for c in cs])


def find_edu_nuclei(vals):
    threshold = filters.threshold_otsu(vals)
    return vals > threshold


def find_edu_nuclei_in_df(df, 
                          image_col='sample_name', 
                          intensity_col='intensity_median'):
    df['edu'] = df.groupby(image_col, 
                           group_keys=False)[intensity_col].apply(
        find_edu_nuclei)
    return df


def quantify_path(img_filepath, labeled_filepath):
    labeled = get_labeled(labeled_filepath)
    img_file = get_file(img_filepath)
    img = get_image(img_file)
    scale = get_pixel_size(img_file)

    return quantify(img, labeled, scale)


def quantify(img, labeled, scale, extra_image=None):
    nuclei_props = get_all_props(img, labeled, extra_image=extra_image)
    nuclei_props = calculate_area(nuclei_props, scale)
    nuclei_props = calculate_volume(nuclei_props, scale)

    return nuclei_props

import matplotlib.pyplot as plt
import matplotlib as mpl
import napari
import numpy as np
import pandas as pd
import seaborn as sns
import tifffile as tif


condition_palette = {'Ndr': 'orange', 'WT': 'b'}
size_palette = {'small': '#6495ED', 'medium': '#FCE883', 'big': '#FF6347'}


def save_img(save_path, stack, axes='YX', create_dir=False):
    """Saves stack as 16-bit integer in tif format."""
    stack = stack.astype('int16')

    # Fill array with new axis
    ndims = len(stack.shape)
    while ndims < 5:
        stack = stack[np.newaxis, :]
        ndims = len(stack.shape)

    # Add missing and correct axes order according to fiji
    new_axes = [ax for ax in 'TZXCY' if ax not in axes[:]]
    axes = ''.join(new_axes) + axes

    stack = tif.transpose_axes(stack, axes, asaxes='TZCYX')

    if create_dir and not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    tif.imsave(str(save_path), data=stack, imagej=True)


def relabel(labeled, swap):
    """Takes a labeled mask and a list of tuples of the swapping labels. If a label is not swapped, it will be
    deleted."""
    out = np.zeros_like(labeled)

    for new, old in swap:
        out[labeled == old] = new

    return out


def relabel_by(labeled, nuclei_props, size_proxy='area'):
    return relabel(labeled, list(zip(nuclei_props[size_proxy].values, nuclei_props['label'].values)))


def view_2d(img, labeled):
    viewer = napari.view_image(img)
    viewer.add_labels(labeled)
    napari.run()


def view_3d(img, labeled, scale=(6, 1, 1)):
    viewer = napari.Viewer(ndisplay=3)
    viewer.add_image(img, blending='additive', scale=scale)
    viewer.add_labels(labeled, scale=scale)
    napari.run()


def view_property_2d(img, property_labeled):
    viewer = napari.view_image(img)
    viewer.add_image(property_labeled)
    napari.run()


def plot_area_labeled(area_labeled, z, area_labeled_limits=(1, 400), axs=None):
    cmap = mpl.cm.get_cmap("coolwarm").copy()
    cmap.set_under('white', alpha=0)

    if axs is None:
        fig, axs = plt.subplots(1, 1)
    this_plot = axs.imshow(area_labeled[z], vmin=area_labeled_limits[0], vmax=area_labeled_limits[1], cmap=cmap,
                           interpolation='none', rasterized=True)
    axs.axis('off')
    axs.set_frame_on(True)
    plt.colorbar(this_plot, ax=axs)
    return axs


def plot_labeled(labeled, z, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 1)
    this_plot = axs.imshow(labeled[z], vmin=1, cmap='tab20', interpolation='none', rasterized=True)
    axs.axis('off')
    plt.colorbar(this_plot, ax=axs)
    return axs


def plot_img(img, z, axs=None):
    if axs is None:
        fig, axs = plt.subplots(1, 1)
    this_plot = axs.imshow(img[z], cmap='Greys_r', rasterized=True)
    axs.axis('off')
    plt.colorbar(this_plot, ax=axs)
    return axs


def plot_hist(vals, max_size=1500, bin_width=25, axs=None, normalize=False,
              cumulative=False):
    bins = np.arange(0, max_size, bin_width)
    if axs is None:
        fig, axs = plt.subplots(1, 1)
    plt.hist(vals, bins=bins, log=True, edgecolor='k', density=normalize,
             cumulative=cumulative, rasterized=True)
    plt.xlabel('Area (um^2)')
    ylabel_name = 'Cumulative %' if cumulative else 'Frequency'
    plt.ylabel(ylabel_name)
    plt.grid()
    return axs


def plot_all(img, labeled, area_labeled, vals, z, area_labeled_limits=(1, 400), max_size=1500, bin_width=25):
    fig, axs = plt.subplots(2, 2, figsize=(15, 13))

    plot_img(img, z, axs=axs[0, 0])
    plot_area_labeled(area_labeled, z, area_labeled_limits=area_labeled_limits, axs=axs[0, 1])
    plot_labeled(labeled, z, axs=axs[1, 0])
    plot_hist(vals, max_size=max_size, bin_width=bin_width, axs=axs[1, 1])

    return fig, axs


def load_full_df(excel_path):
    df = pd.read_excel(excel_path, sheet_name=None)
    dfs = []
    for sample, this_df in df.items():
        if len(this_df) == 0:
            continue
        this_df['sample_name'] = sample
        this_df['condition'] = 'Ndr' if sample[:2] == 'Nd' else 'WT'
        this_df['region'] = 'Hilar' if 'hilar' in sample.lower() else 'Periphery'
        this_df['line'] = sample[3:7] if sample[:2] == 'Nd' else sample[2:6]
        dfs.append(this_df)
    return pd.concat(dfs, ignore_index=True)


def plot_hist_per_sample(df, max_bin=800, bin_width=25):
    g = sns.FacetGrid(df, hue="condition", col="sample_name",
                      col_wrap=np.floor(np.sqrt(len(df.sample_name.unique()))).astype(int),
                      sharex=True, palette=condition_palette)
    g.map(sns.histplot, "area", alpha=.4, binrange=(0, max_bin),
          binwidth=bin_width, stat='count', edgecolor='k', kde=True, log=True,
          rasterized=True)
    g.set(ylim=(0.9, 5000), xlim=(-10, max_bin))
    g.map(plt.grid, color='grey', alpha=0.6, axis='y')
    return g


def plot_full_size_hist(df):
    fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
    sns.histplot(df, hue='condition', x='area', log_scale=True, palette=condition_palette, ax=axs[0])
    sns.histplot(df, x='area', log_scale=True, ax=axs[1])
    plt.subplots_adjust(hspace=0)
    for ax in axs:
        ax.grid('on')
    return fig, axs


def plot_normalized_size_hist(df, log_y=False):
    fig, axs = plt.subplots(2, 1, figsize=(5, 8), sharex=True)
    sns.histplot(df, hue='condition', x='area', log_scale=(True, log_y), stat="density",
                 common_norm=False, palette=condition_palette, ax=axs[0])
    sns.histplot(df, hue='condition', x='area', log_scale=(True, log_y), stat="density",
                 common_norm=False, palette=condition_palette, cumulative=True,
                 element="step", fill=False,
                 ax=axs[1])
    plt.subplots_adjust(hspace=0)
    for ax in axs:
        ax.grid('on')
    return fig, axs


def plot_classification(df, lines):
    hue_order = ['small', 'medium', 'big'] if len(lines) == 2 else ['small', 'big']
    axs = sns.histplot(df, hue='predicted', x='area', log_scale=True, palette=size_palette,
                       hue_order=hue_order)
    for line_name, line in lines:
        try:
            axs.axvline(x=line, color='k', linestyle='--', alpha=0.9)
            axs.text(line, 400, str(line), fontsize=10)
        except ValueError:
            pass
    return axs


def plot_stackedbar_classification(df, groupby='sample_name', normalize=True):
    level_depth = 1 if isinstance(groupby, str) else 2
    cols = ['small', 'medium', 'big'] if 'medium' in df.predicted.values else ['small', 'big']
    axs = df.groupby(groupby).predicted.value_counts(normalize=normalize).unstack(level=level_depth)[cols].plot(
        kind='bar', stacked=True, edgecolor='k', log=not normalize)
    axs.grid('on', axis='y')
    if not normalize:
        axs.set_ylim(ymin=0.92)
    return axs

from itertools import repeat
import pathlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import sys
from tqdm import tqdm

from nuclei_segmenter.loader import get_file, get_image, get_labeled, get_pixel_size
from nuclei_segmenter.quantify import get_good_planes, get_gmm, classify, get_separation, unify_labels_by_area
from nuclei_segmenter.vis import relabel_by, plot_all, plot_hist, load_full_df, plot_hist_per_sample, \
    plot_full_size_hist, plot_classification, plot_stackedbar_classification


DATA_DIR = pathlib.Path(r'P:\c5_Andersson_Emma\Afshan\Nuclei Area quantication')
SEGM_DIR = pathlib.Path(r'P:\c5_Andersson_Emma\Afshan\Nuclei Area quantication\segmented_nuclei\segmented')
file_list = [filepath for filepath in SEGM_DIR.iterdir() if filepath.is_file()
             and filepath.suffix == '.tiff']

area_labeled_limits = (1, int(sys.argv[1]))
max_size = int(sys.argv[2])
bin_width = int(sys.argv[3])


def small_report(segment_path, excel_writer):
    print('Preparing ' + segment_path.name)
    filepath = DATA_DIR / (segment_path.stem + '.czi')
    prop_path = segment_path.with_name(filepath.stem + '.xlsx')

    file = get_file(filepath)

    print('Loading image')
    image = get_image(file)
    scale = get_pixel_size(file)
    labels = get_labeled(segment_path)
    nuclei_props = pd.read_excel(prop_path)
    nuclei_props = unify_labels_by_area(nuclei_props)

    print('Relabeling image')
    area_labeled = relabel_by(labels, nuclei_props)

    print('Estimating good planes')
    good_planes = get_good_planes(labels, threshold=0.7)

    print('Saving report at ' + str(segment_path.with_suffix('.pdf')))
    with PdfPages(segment_path.with_suffix('.pdf')) as pp:
        for plane in good_planes:
            plot_all(image, labels, area_labeled, nuclei_props[nuclei_props['z'] == plane].area.values, plane,
                     area_labeled_limits=area_labeled_limits, max_size=max_size, bin_width=bin_width)
            pp.savefig()
            plt.close()

        nuclei_props = nuclei_props[nuclei_props['z'].isin(good_planes)].query('area > 10')
        plot_hist(nuclei_props.area.values, max_size=max_size, bin_width=bin_width)
        pp.savefig()
        plt.close()

        print('Saving big excel file')
        nuclei_props.to_excel(excel_writer, sheet_name=segment_path.stem)


def summary_report(nuclei_size_excel_path):
    df = load_full_df(nuclei_size_excel_path)
    df_2 = df.copy()

    model = get_gmm(df.area.values)
    df['predicted'] = classify(model, df.area.values)

    with PdfPages(nuclei_size_excel_path.with_name('summary_3_classes.pdf')) as pp:
        add_plots_to_pdf(df, model, pp)

    with pd.ExcelWriter(nuclei_size_excel_path.with_name('summary_3_classes.xlsx')) as writer:
        save_summary_tables(df, writer)


    model_2 = get_gmm(df.area.values, number_of_classes=2)
    df_2['predicted'] = classify(model_2, df.area.values)

    with PdfPages(nuclei_size_excel_path.with_name('summary_2_classes.pdf')) as pp:
        add_plots_to_pdf(df_2, model_2, pp)

    with pd.ExcelWriter(nuclei_size_excel_path.with_name('summary_2_classes.xlsx')) as writer:
        save_summary_tables(df_2, writer)


def save_summary_tables(df, writer):
    cols = ['small', 'medium', 'big'] if 'medium' in df.predicted.values else ['small', 'big']
    df.groupby('sample_name').predicted.value_counts(
        normalize=False).unstack(
        level=1)[cols].to_excel(writer, sheet_name='counts per sample')
    df.groupby('sample_name').predicted.value_counts(
        normalize=True).unstack(
        level=1)[cols].to_excel(writer, sheet_name='normalized per sample')
    df.groupby('condition').predicted.value_counts(
        normalize=False).unstack(
        level=1)[cols].to_excel(writer, sheet_name='counts per condition')
    df.groupby('condition').predicted.value_counts(
        normalize=True).unstack(
        level=1)[cols].to_excel(writer, sheet_name='normalized per condition')


def add_plots_to_pdf(df, model, pp):
    plot_hist_per_sample(df)
    pp.savefig()
    plt.close()

    plot_full_size_hist(df)
    pp.savefig()
    plt.close()

    plot_classification(df, get_separation(model))
    plt.tight_layout()
    pp.savefig()
    plt.close()

    plot_stackedbar_classification(df, 'sample_name')
    plt.tight_layout()
    pp.savefig()
    plt.close()

    plot_stackedbar_classification(df, 'condition')
    plt.tight_layout()
    pp.savefig()
    plt.close()

    plot_stackedbar_classification(df, 'sample_name', normalize=False)
    plt.tight_layout()
    pp.savefig()
    plt.close()

    plot_stackedbar_classification(df, 'condition', normalize=False)
    plt.tight_layout()
    pp.savefig()
    plt.close()


if __name__ == '__main__':
    # with Pool(15) as p:
    #     for this in tqdm(p.imap_unordered(run_and_save, file_list)):
    #         pass
    with pd.ExcelWriter(SEGM_DIR.parent / 'nuclei_size.xlsx') as writer:
        for this in tqdm(map(small_report, file_list, repeat(writer)), total=len(file_list)):
            pass

    summary_report(SEGM_DIR.parent / 'nuclei_size.xlsx')

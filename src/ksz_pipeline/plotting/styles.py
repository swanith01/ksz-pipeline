"""
Matplotlib style dicts and save helpers shared across both pipelines.

Extracted from the top of 16Jun2026_copy_PatchyScreening_SkewedLOS_LightconeKSZ.py.
Identical style is used in 28May2026_kSZBoxes.py.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

PDF_STYLE = {
    'font.family'        : 'serif',
    'font.serif'         : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset'   : 'cm',
    'font.size'          : 30,
    'axes.labelsize'     : 29,
    'axes.titlesize'     : 40,
    'xtick.labelsize'    : 30,
    'ytick.labelsize'    : 30,
    'legend.fontsize'    : 20,
    'figure.titlesize'   : 28,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.top'          : True,
    'ytick.right'        : True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'xtick.major.size'   : 6,
    'ytick.major.size'   : 6,
    'xtick.minor.size'   : 3,
    'ytick.minor.size'   : 3,
    'xtick.major.width'  : 1.0,
    'ytick.major.width'  : 1.0,
    'xtick.minor.width'  : 0.8,
    'ytick.minor.width'  : 0.8,
    'axes.linewidth'     : 1.0,
    'lines.linewidth'    : 1.8,
    'lines.markersize'   : 5,
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
}

PNG_STYLE = {
    'font.family'        : 'serif',
    'font.serif'         : ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset'   : 'cm',
    'font.size'          : 15,
    'axes.labelsize'     : 25,
    'axes.titlesize'     : 18,
    'xtick.labelsize'    : 25,
    'ytick.labelsize'    : 25,
    'legend.fontsize'    : 18,
    'figure.titlesize'   : 15,
    'xtick.direction'    : 'in',
    'ytick.direction'    : 'in',
    'xtick.top'          : True,
    'ytick.right'        : True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'axes.linewidth'     : 1.0,
    'lines.linewidth'    : 1.5,
    'figure.dpi'         : 150,
    'savefig.dpi'        : 300,
    'savefig.bbox'       : 'tight',
    'savefig.pad_inches' : 0.05,
}


def save_pdf_png(plot_func, plot_dir, plot_name, title=None, figsize=(10, 7)):
    """Save a single-axis plot as both PDF and PNG."""
    with mpl.rc_context(PDF_STYLE):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_func(ax)
        ax.set_title("")
        fig.savefig(f"{plot_dir}/{plot_name}.pdf")
        plt.close(fig)

    with mpl.rc_context(PNG_STYLE):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_func(ax)
        if title is not None:
            ax.set_title(title, fontweight='bold')
        fig.savefig(f"{plot_dir}/{plot_name}.png")
        plt.close(fig)


def save_pdf_png_multi(plot_func, plot_dir, plot_name, title=None,
                       figsize=(20, 4.2)):
    """Save a multi-panel plot (plot_func receives the whole figure)."""
    with mpl.rc_context(PDF_STYLE):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        plot_func(fig)
        fig.savefig(f"{plot_dir}/{plot_name}.pdf")
        plt.close(fig)

    with mpl.rc_context(PNG_STYLE):
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        plot_func(fig)
        if title is not None:
            fig.suptitle(title, fontweight='bold')
        fig.savefig(f"{plot_dir}/{plot_name}.png")
        plt.close(fig)

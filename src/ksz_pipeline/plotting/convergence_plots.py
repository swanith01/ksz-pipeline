"""
Convergence summary plots. No science logic here (per this repo's
convention -- see README) -- purely rendering, given already-computed
D3000 values.

NOTE: does not use plotting.styles (PNG_STYLE/PDF_STYLE/save_pdf_png,
imported by the exploratory notebooks) because that module's contents
haven't been provided -- plain matplotlib defaults used instead. Restyle
once styles.py is available, for consistency with the rest of the
project's figures.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_d3000_convergence(param_values, d3000_dict, xlabel, title, out_path,
                            reference_value=None, reference_label=None):
    """
    Plot D_3000 vs a swept parameter (box size or HII_DIM), for one or
    more D_ell "flavors" (e.g. direct/Cain vs Georgiev) on the same axes.

    Parameters
    ----------
    param_values : sequence of float, the swept parameter's values,
                   ascending, matching the keys' order in d3000_dict
    d3000_dict   : dict {flavor_label: [D3000 values, same order/length
                   as param_values]} -- one line per flavor
    xlabel       : str
    title        : str
    out_path     : str, saved as out_path + '.png' (no .pdf -- add if
                   wanted once styles.py is available)
    reference_value : float, optional -- draws a horizontal reference
                   line (e.g. the fiducial 800Mpc/128^3 value once known)
    reference_label  : str, optional
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, values in d3000_dict.items():
        ax.plot(param_values, values, 'o-', lw=2.0, ms=7, label=label)

    if reference_value is not None:
        ax.axhline(reference_value, color='gray', ls='--', lw=1.2,
                    label=reference_label or 'reference')
        ax.axhspan(reference_value * 0.95, reference_value * 1.05,
                    color='gray', alpha=0.12)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(r'$D_{3000}\ [\mu K^2]$')
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path + '.png', dpi=200)
    plt.close(fig)

"""
Convergence sweep parameter-list builders.

Shared by scripts/04_convergence_coeval.py and
scripts/05_convergence_stitched.py so the box-size/resolution sweep
definitions can't drift between the two methods -- both need to sweep
the exact same (BOX_LEN, HII_DIM) configurations for the comparison to
mean anything.
"""


def build_param_list(sweep, conv_cfg):
    """
    Build the (BOX_LEN, HII_DIM, tag) list for the requested sweep type.

    boxsize    : dx fixed at conv_cfg['fixed_dx_mpc'], BOX_LEN varies
                 over conv_cfg['box_lens'], HII_DIM = round(L/dx) per box
    resolution : BOX_LEN fixed at conv_cfg['ref_box_len'], HII_DIM
                 varies over conv_cfg['hii_dims']

    Parameters
    ----------
    sweep    : 'boxsize' or 'resolution'
    conv_cfg : dict, the config's `convergence:` section -- needs
               fixed_dx_mpc + box_lens for 'boxsize', or ref_box_len +
               hii_dims for 'resolution'

    Returns
    -------
    param_list : list of (BOX_LEN, HII_DIM, tag)
    xlabel     : str, for convergence plots
    x_values   : list, the swept parameter's values (BOX_LEN for
                 'boxsize', HII_DIM for 'resolution'), same order as
                 param_list
    """
    if sweep == 'boxsize':
        dx = conv_cfg['fixed_dx_mpc']
        param_list = []
        for L in conv_cfg['box_lens']:
            N = max(4, int(round(L / dx)))
            param_list.append((float(L), N, f"box{L}_N{N}"))
        xlabel = r'Box size $L$ [Mpc]'
        x_values = [p[0] for p in param_list]
    elif sweep == 'resolution':
        L = conv_cfg['ref_box_len']
        param_list = [(float(L), int(N), f"res{N}") for N in conv_cfg['hii_dims']]
        xlabel = r'$HII\_DIM$'
        x_values = [p[1] for p in param_list]
    else:
        raise ValueError(f"Unknown sweep type: {sweep}")
    return param_list, xlabel, x_values

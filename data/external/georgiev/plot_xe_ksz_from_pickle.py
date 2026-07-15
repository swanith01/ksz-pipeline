import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, cm, colors

rc('font', **{'family': 'serif', 'serif': ['times new roman'], 'size': 14})
rc('text', usetex=False)
rc('axes', linewidth=1.5)


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap


with open('obs_vs_params_xe_ksz_data.pickle', 'rb') as f:
    data = pickle.load(f)

zlin = data['zlin']
ndeg = data['ndeg']
ntest = data['ntest']
params_test = data['params_test']
labels = data['labels']
paramnames = data['paramnames']
results = data['results']
lrange = data['lrange']

# 3 columns: colorbar, xe(z), ksz
fig, axes = plt.subplots(ndeg, 3, figsize=(9, 10),
                          sharex='col', sharey='col',
                          gridspec_kw={'width_ratios': (1, 3, 3)})
fig.subplots_adjust(left=0.05, wspace=0.35, hspace=0.04, top=0.98, right=0.98, bottom=0.06)

for i in range(ndeg):

    cmap = truncate_colormap(cm.get_cmap('PuRd'), 0.3, 1.0)
    norm = colors.Normalize(vmin=np.min(params_test[:, i]), vmax=np.max(params_test[:, i]))

    axes[i, 0].set_visible(False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    c_bar = plt.colorbar(sm, ax=axes[i, 0], shrink=.85, fraction=.7, aspect=10, label=labels[i])
    c_bar.ax.yaxis.set_label_position('left')

    for v in range(axes.shape[-1]):
        axes[i, v].grid()

    if i == ndeg - 1:
        axes[i, 1].set_xlabel(r"Redshift $z$")
        axes[i, 2].set_xlabel(r"Angular multipole $\ell$")
    axes[i, 2].set_ylabel(r"$\mathcal{D}_\ell^\mathrm{kSZ}$ [$\mu$K$^2$]")
    axes[i, 1].set_ylabel(r"$x_e(z)$")

    for u in range(ntest):
        theta = results[i]['theta'][u]
        xe_curve = results[i]['xe'][u]
        model_ksz = results[i]['ksz'][u]

        # XE
        axes[i, 1].plot(zlin, xe_curve, color=cmap(norm(theta[i])), lw=1.5)

        # KSZ
        axes[i, 2].plot(lrange, model_ksz, color=cmap(norm(theta[i])), lw=1.5)

fig.tight_layout()
fig.savefig('obs_vs_params_xe_ksz.pdf')
print('Saved figure to obs_vs_params_xe_ksz.pdf')

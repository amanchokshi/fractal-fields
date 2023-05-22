"""Example Plots with Fractal Fields"""

import matplotlib.pyplot as plt
import cmasher as cmr
import numpy as np

from grf import FractalField

nice_fonts = {
    # Use LaTeX to write all text
    "text.usetex": True,
    "font.family": "serif",
    "axes.labelsize": 12,
    "font.size": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
}

plt.rcParams.update(nice_fonts)


def add_colorbar(mappable):
    """Helper function to add colorbar to plot."""

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    import warnings
    warnings.filterwarnings("ignore")

    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.set_yticklabels(["{:.3f}".format(i) for i in cbar.get_ticks()])
    plt.sca(last_axes)
    return cbar


if __name__ == "__main__":

    ###################################
    # PLOT A SINGLE 2D FIELD          #
    ###################################
    # ff = FractalField(
    #     size_exp=10,
    #     dim=2,
    #     pix_scale=1,
    #     amp=1,
    #     alpha=2,
    #     seed=42
    # )
    # ff.generate_field()
    # ff.ps()
    #
    # ymin_pow = np.floor(np.log10(min(ff.pk)))
    # ymax_pow = np.ceil(np.log10(max(ff.pk)))
    # y_ticks = [10**i for i in np.arange(ymin_pow, ymax_pow + 1)]
    #
    # fig, ax1 = plt.subplots(1, 1, figsize=(7, 7))
    # im = ax1.pcolormesh(
    #     ff.xgrid[0], ff.xgrid[1],
    #     ff.x_field.real, cmap='cmr.wildfire'
    # )
    #
    # title_str = fr"""GRF Realization {{{(ff.size,)*ff.dim}}}:
    #         $P(k)={{{ff.amp}}} \times k^{{{-1*ff.alpha}}}$"""
    #
    # ticks = [0, 2**(ff.size_exp-1), 2**ff.size_exp]
    # tick_labels = ['0', rf'$2^{{{ff.size_exp - 1}}}$',
    #                rf'$2^{{{ff.size_exp}}}$']
    #
    # ax1.set_title(title_str)
    # ax1.set_xticks(ticks, tick_labels)
    # ax1.set_yticks(ticks, tick_labels)
    #
    # ax1.set_xlabel(r'x [$pixels$]')
    # ax1.set_ylabel(r'y [$pixels$]')
    # add_colorbar(im)
    # plt.tight_layout()
    # plt.savefig('imgs/GRF_2D.png')
    # plt.show()

    ###################################
    # PLOT ENSEMBLE OF 1D PS          #
    ###################################

    seeds = np.arange(256)

    kvals = []
    pks = []
    for seed in seeds:
        ff = FractalField(
            size_exp=10,
            dim=2,
            pix_scale=1,
            amp=1,
            alpha=0,
            seed=seed
        )
        ff.generate_field()
        ff.ps()

        kvals.append(ff.kvals)
        pks.append(ff.pk)

    kvals = np.asarray(kvals[0])
    pks = np.asarray(pks)
    pks_mean = np.mean(pks, axis=0)

    # ymin_pow = np.floor(np.log10(min(pks_mean)))
    # ymax_pow = np.ceil(np.log10(max(pks_mean)))
    # y_ticks = [10**i for i in np.arange(ymin_pow, ymax_pow + 1)]

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    title_str = fr"""1D PS of GRF Realization:
            $P(k)={{{ff.amp}}} \times k^{{{-1*ff.alpha}}}$"""

    ax.set_title(title_str)
    ax.loglog(kvals, pks_mean, color="#DA3752",
              label='Average 1D PS', zorder=777, lw=2)
    ax.loglog(ff.kvals, ff.amp*np.power(ff.kvals, -1*ff.alpha, dtype='float'),
              color="#3287BC", label='Fiducial 1D PS', zorder=666, lw=2)
    for i in seeds:
        ax.loglog(kvals, pks[i], color='k', alpha=0.3, lw=0.3)
    ax.set_xlabel('$k [pixel~{-1}]$')
    # ax.set_yticks(y_ticks)
    ax.set_yticks([1/2, 1, 2])
    ax.set_ylim(1/2, 2)
    ax.set_ylabel('$p(k)$')

    leg = ax.legend(loc="upper right", frameon=True,
                    markerscale=4, handlelength=1)
    leg.get_frame().set_facecolor("white")
    for le in leg.legend_handles:
        le.set_alpha(1)
    plt.tight_layout()
    plt.savefig('imgs/GRF_PS_1D.png')
    plt.show()

    ###################################
    # PLOT A RANGE OF SPECTRAL SLOPES #
    ###################################
    # n = 1024
    # d = 2
    # amp = 1
    # p_step = 0.02
    # alphas = np.arange(0, 4+p_step, p_step)
    # for i, alpha in enumerate(tqdm(alphas)):
    #     k_field, x_field, kgrid, xgrid = generate_field(
    #         n=n,
    #         d=d,
    #         pix_scale=1,
    #         amp=amp,
    #         alpha=alpha,
    #         seed=42
    #     )
    #     pk, pk_std, kvals = ps(k_field, kgrid)
    #
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    #     im = ax1.pcolormesh(xgrid[0], xgrid[1],
    #                         x_field.real, cmap='cmr.wildfire')
    #     ax1.set_title(
    #         fr'GRF Realization {{{(n,)*d}}}: $P(k)={{{amp}}} \times k^{{{-1*alpha:.2f}}}$'
    #     )
    #     ax1.set_xlabel(r'x [$pixels$]')
    #     ax1.set_ylabel(r'y [$pixels$]')
    #     ax1.set_xticks([0, 2**9, 2**10],
    #                    ['0', r'$2^9$', r'$2^{10}$'],)
    #     ax1.set_yticks([0, 2**9, 2**10],
    #                    ['0', r'$2^9$', r'$2^{10}$'],)
    #     add_colorbar(im)
    #
    #     ax2.set_title('1D PS of White Noise')
    #     ax2.loglog(kvals, pk, color="#DA3752", label='Measured 1D PS')
    #     ax2.loglog(kvals, amp * np.power(kvals, -1*alpha, dtype='float'),
    #                color="#5E4FA1", label='Fiducial 1D PS', ls=':')
    #     ax2.set_xlabel('$k [pixel~{-1}]$')
    #     ymin_pow = math.floor(math.log10(min(pk)))
    #     ymax_pow = math.ceil(math.log10(max(pk)))
    #     y_ticks = [10**i for i in range(ymin_pow, ymax_pow + 1)]
    #     ax2.set_yticks(y_ticks)
    #     ax2.set_ylabel('$p(k)$')
    #
    #     leg = ax2.legend(loc="upper right", frameon=True,
    #                      markerscale=4, handlelength=1)
    #     leg.get_frame().set_facecolor("white")
    #     for le in leg.legendHandles:
    #         le.set_alpha(1)
    #     plt.tight_layout()
    #     plt.savefig(f'pk/slope/slice_{i:0>3}.jpg', dpi=240)
    #     plt.close()

    ###################################
    # PLOT SLICES OF A 3D CUBE        #
    ###################################
    # n = 256
    # d = 3
    # amp = 1
    # alpha = 3.5
    # k_field, x_field, kgrid, xgrid = generate_field(
    #     n=n,
    #     d=d,
    #     pix_scale=1,
    #     amp=amp,
    #     alpha=alpha,
    #     seed=42
    # )
    # pk, pk_std, kvals = ps(k_field, kgrid)
    #
    # ymin_pow = math.floor(math.log10(min(pk)))
    # ymax_pow = math.ceil(math.log10(max(pk)))
    # y_ticks = [10**i for i in range(ymin_pow, ymax_pow + 1)]
    #
    # pk_2ds = []
    # for i in tqdm(range(n)):
    #
    #     xgrid_i = [xgrid[0][:, :, i], xgrid[1][:, :, i]]
    #     kgrid_i = [kgrid[0][:, :, i], kgrid[1][:, :, i]]
    #     x_field_i = x_field[:, :, i]
    #     k_field_i = k_field[:, :, i]
    #
    #     # fig = plt.figure(figsize=(7, 7))
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    #     im = ax1.pcolormesh(xgrid_i[0], xgrid_i[1],
    #                         x_field_i.real, cmap='cmr.wildfire', vmin=-0.02, vmax=0.02)
    #     ax1.set_title(
    #         fr'GRF Realization {{{(n,)*d}}}: $P(k)={{{amp}}} \times k^{{{-1*alpha}}}$'
    #     )
    #     ax1.set_xlabel(r'x [$pixels$]')
    #     ax1.set_ylabel(r'y [$pixels$]')
    #     add_colorbar(im)
    #
    #     pk_i, pk_std_i, kvals_i = ps(np.fft.ifft(x_field_i), kgrid_i)
    #     pk_2ds.append(pk_i)
    #
    #     ax2.set_title('1D PS of White Noise')
    #     ax2.loglog(kvals, pk, color="#3287BC",
    #                label='Spherically Averaged PS', lw=2, zorder=666)
    #     ax2.loglog(kvals, pk_i, color="#DA3752",
    #                label='Slice Averaged PS', zorder=444)
    #     for i, p in enumerate(pk_2ds):
    #         ax2.loglog(kvals, p, color="#DA3752",
    #                    lw=0.7, alpha=0.7, zorder=333)
    #     ax2.loglog(kvals, amp*np.power(kvals, -1*alpha, dtype='float'),
    #                color="#5E4FA1", label='Fiducial PS', ls=':', lw=2, zorder=555)
    #
    #     ax2.set_xlabel('$k [pixel~{-1}]$')
    #     ax2.set_yticks(y_ticks)
    #     ax2.set_ylabel('$p(k)$')
    #
    #     leg = ax2.legend(loc="upper right", frameon=True,
    #                      markerscale=4, handlelength=1)
    #     leg.get_frame().set_facecolor("white")
    #     for le in leg.legendHandles:
    #         le.set_alpha(1)
    #     plt.tight_layout()
    #     # plt.show()
    #     plt.savefig(f'pk/slice/slice_{i:0>3}.jpg', dpi=240)
    #     plt.close()

    # # Slices of 3D realization
    # alpha = 3.5
    # size_expo = 9
    # shape = (2**size_expo, 2**size_expo, 2**size_expo)
    # field = generate_field(alpha, shape)
    #
    # for i in tqdm(range(2**size_expo)):
    #     fig = plt.figure(figsize=(7, 7))
    #     # plt.title(r'Realization in k-space of given P(k)')
    #     im = plt.imshow(field[:, :, i], cmap='cmr.wildfire',
    #                     vmax=0.007, vmin=-0.007)
    #     plt.xticks([0, 2**(size_expo-1), 2**size_expo],
    #                ['0', rf'$2^{{{size_expo - 1}}}$', rf'$2^{{{size_expo}}}$'],)
    #     plt.yticks([0, 2**(size_expo-1), 2**size_expo],
    #                ['0', rf'$2^{{{size_expo - 1}}}$', rf'$2^{{{size_expo}}}$'],)
    #     plt.title(rf'Slice [{i}]: $P(k) \propto k^{{{-1*alpha:.2f}}}$')
    #     add_colorbar(im)
    #     plt.tight_layout()
    #     # plt.show()
    #     plt.savefig(f'pk/k35/slice_{i:0>3}.jpg', dpi=240)
    #     plt.close()

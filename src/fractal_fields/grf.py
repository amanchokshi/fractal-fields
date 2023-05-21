"""
Generate a Gaussian Random Field with a given power-law Power Spectrum.

Based on: https://garrettgoon.com/gaussian-fields
"""

import numpy as np
import scipy.stats as stats


class FractalField:
    r"""
    Create real and fourier space realization of Gaussian Random Fields
    with a given power law power spectrum

    The power spectum takes the form:

        P(k) = A * k ^ ( -1 * ⍺ )

    Parameters
    ----------
    size_exp: int, default 8
        simulation size is defined at 2**siz_exp

    dim: int, default 2
        simulation dimensions

    pix_scale: float, default 1
        How much physical length a pixel represents,
        in physical units / pixel

    amp: float, default 1
        amplitude [A] of the generating
        power spectrum power law

    alpha: float, default 2
        power law spectral index [⍺] described above

    seed: int, default 42
        seed of the random number generator,
        to make things reproducible

    """

    def __init__(
            self,
            size_exp=8,
            dim=2,
            pix_scale=1,
            amp=1,
            alpha=2,
            seed=42
    ):
        self.size_exp = size_exp
        self.size = 2**self.size_exp
        self.dim = dim
        self.pix_scale = pix_scale
        self.amp = amp
        self.alpha = alpha
        self.seed = seed

        # Seed random number generator
        self.rng = np.random.default_rng(self.seed)

    @property
    def kvector(self):
        """Wavenumber k for one side."""
        return np.fft.fftfreq(self.size, d=self.pix_scale)

    @property
    def kgrid(self):
        """Wavenumber grid for simulation cube."""
        return np.meshgrid(
            *np.tile(self.kvector, (self.dim, 1)), indexing="ij"
        )

    @property
    def knorm(self):
        """Radial k distance for every point in kgrid"""
        return np.sqrt(np.sum(np.power(self.kgrid, 2), axis=0))

    @property
    def xvector(self):
        """Position x for one side."""
        return np.arange(self.size * self.pix_scale)

    @property
    def xgrid(self):
        """Position grid for simulation cube."""
        return np.meshgrid(
            *np.tile(self.xvector, (self.dim, 1)), indexing="ij"
        )

    def white_noise(self):
        """Generate unit normal white noise.

        White noise with spectrally uniform power is generated 
        in real space, with completely real values. This results
        in a k-space realization which is hermitian.

        We have to normalize the k-space representation by a factor of

            C = 1 / N^{d/2}
        """
        white_x = self.rng.normal(size=(self.size,)*self.dim)
        return np.fft.fftn(white_x) / (np.power(self.size, self.dim/2))

    def generate_field(self):
        """Generate real and k-space realization of GRF"""

        # Regulate the k=0 divergence
        # This step results in a mean zero field
        knorm = self.knorm
        mask = self.knorm > 0

        # Empty array for spectral power
        k_field = np.zeros_like(knorm)

        # Generate the PS with given spectral index alpha
        spec_func = self.amp * \
            np.power(knorm[mask], -1 * self.alpha, dtype='float')

        k_field[mask] = np.sqrt(spec_func)

        # This is the realization of a GRF with a given PS in k space
        # white noise * sqrt(desired PS spectra)
        self.k_field = self.white_noise() * k_field
        self.x_field = np.fft.ifftn(self.k_field)

    def ps(self):
        """Generate 1D Power Spectrum of given field."""

        # Amplitude of power spectrum in original dimensions
        ps_amps = np.abs(self.k_field)**2

        # max k value that can be sampled
        k_max = 1 / self.pix_scale

        # k resolution
        dk = 1 / (self.kvector.shape[0] * self.pix_scale)

        # k bin start & stop values
        kbins = np.arange(dk/2, k_max/2 + dk, dk)

        # k bin mean
        self.kvals = 0.5 * (kbins[1:] + kbins[:-1])

        self.pk, _, _ = stats.binned_statistic(
            self.knorm.flatten(), ps_amps.flatten(),
            statistic="mean",
            bins=kbins)

        self.pk_std, _, _ = stats.binned_statistic(
            self.knorm.flatten(), ps_amps.flatten(),
            statistic="std",
            bins=kbins)


if __name__ == "__main__":

    import math
    # from tqdm import tqdm
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # beautiful colormap
    # import wildfire
    import cmasher as cmr

    nice_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        "axes.labelsize": 10,
        "font.size": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
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

    ###################################
    # PLOT A SINGLE 2D FIELD          #
    ###################################
    ff = FractalField(
        size_exp=10,
        dim=2,
        pix_scale=1,
        amp=1,
        alpha=2,
        seed=42
    )
    ff.generate_field()
    ff.ps()

    ymin_pow = math.floor(math.log10(min(ff.pk)))
    ymax_pow = math.ceil(math.log10(max(ff.pk)))
    y_ticks = [10**i for i in range(ymin_pow, ymax_pow + 1)]

    # fig = plt.figure(figsize=(7, 7))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    im = ax1.pcolormesh(ff.xgrid[0], ff.xgrid[1],
                        ff.x_field.real, cmap='cmr.wildfire')
    ax1.set_title(
        fr'GRF Realization {{{(ff.size,)*ff.dim}}}: $P(k)={{{ff.amp}}} \times k^{{{-1*ff.alpha}}}$'
    )
    ax1.set_xlabel(r'x [$pixels$]')
    ax1.set_ylabel(r'y [$pixels$]')
    add_colorbar(im)

    ax2.set_title('1D PS of White Noise')
    ax2.loglog(ff.kvals, ff.pk, color="#DA3752", label='Measured 1D PS')
    ax2.loglog(ff.kvals, ff.amp*np.power(ff.kvals, -1*ff.alpha, dtype='float'),
               color="#5E4FA1", label='Fiducial 1D PS', ls=':')
    ax2.set_xlabel('$k [pixel~{-1}]$')
    ax2.set_yticks(y_ticks)
    ax2.set_ylabel('$p(k)$')

    leg = ax2.legend(loc="upper right", frameon=True,
                     markerscale=4, handlelength=1)
    leg.get_frame().set_facecolor("white")
    for le in leg.legendHandles:
        le.set_alpha(1)
    plt.tight_layout()
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

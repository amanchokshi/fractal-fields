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

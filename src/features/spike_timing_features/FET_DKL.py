import numpy as np
import scipy.stats as stats


class DKL(object):
    """
    This feature compares the CDF of the initial part of the histogram to a uniform CDF
    using the D_kl metric.
    """

    def __init__(self):
        self.name = 'D_KL'

    @staticmethod
    def calculate_feature(**kwargs):
        """
        inputs:
        start_cdf: One dimensional ndarray. Starting part of the cumulative distribution function
        kwargs: Can be ignored, used only for compatibility

        returns:
        Calculated feature value as described before.
        """
        try:
            start_band = kwargs['start_band']
            mid_band = kwargs['mid_band']
            assert start_band is not None and mid_band is not None
        except KeyError:
            raise TypeError

        start_band_dens = (start_band.T / np.sum(start_band, axis=1)).T
        uniform = np.ones(start_band.shape[1]) / start_band.shape[1]

        result = np.zeros((len(start_band), 2))

        for i, dens in enumerate(start_band_dens):
            dkl = stats.entropy(dens, uniform)
            if dkl == float('inf'):
                print(dens)
                raise AssertionError
            result[i, 0] = dkl

        mid_dens = (mid_band.T / np.sum(mid_band, axis=1)).T
        uniform = np.ones(mid_dens.shape[1]) / mid_dens.shape[1]

        for i, dens in enumerate(mid_dens):
            dkl = stats.entropy(dens, uniform)
            if dkl == float('inf'):
                print(dens)
                raise AssertionError
            result[i, 1] = dkl

        return result

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f"{self.name}_short", f"{self.name}_long"]

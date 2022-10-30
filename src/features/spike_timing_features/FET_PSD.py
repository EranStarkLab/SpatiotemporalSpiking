import numpy as np
import scipy.signal as signal
from constants import UPSAMPLE, ACH_WINDOW, ACH_RESOLUTION


class PSD(object):
    """
    This feature performs power spectral analysis on the histogram calculating the centroid of the power spectral
    density and the centroid of its derivative
    """

    def __init__(self, resolution=ACH_RESOLUTION, mid_band_end=ACH_WINDOW):
        self.resolution = resolution
        self.mid_band_end = mid_band_end

        self.name = 'PSD'

    def calculate_feature(self, **kwargs):
        """
        inputs:
        rhs: One dimensional ndarray. Right hand side of the histogram, used for calculation of the start_cdf if not provided
        kwargs: Can be ignored, used only for compatibility

        returns:
        Calculated measurements of the feature value as described before.
        """
        try:
            rhs = kwargs['rhs']
            assert rhs is not None
        except KeyError:
            raise TypeError

        result = np.zeros((len(rhs), 2))
        fs = self.mid_band_end * self.resolution * UPSAMPLE
        for i, rh in enumerate(rhs):
            rh = rh - rh.mean()
            f, pxx = signal.periodogram(rh, fs)
            inds = (f <= 100) * (f > 0)
            f, pxx = f[inds], pxx[inds]
            centroid = np.sum(f * pxx) / np.sum(pxx)

            der_pxx = np.abs(np.gradient(pxx))
            der_centroid = np.sum(f * der_pxx) / np.sum(der_pxx)

            result[i, 0] = centroid
            result[i, 1] = der_centroid

        return result

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f"{self.name}_center", f"{self.name}'_center"]

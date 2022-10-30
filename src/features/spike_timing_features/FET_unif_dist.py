import numpy as np


class UnifDist(object):
    """
    This feature compares the starting band cumulative distribution function of the histogram to a linear (uniform)
    change.
    """

    def __init__(self):
        self.name = 'Uniform_distance'

    @staticmethod
    def calculate_feature(**kwargs):
        """
        inputs:
        start_cdf: One dimensional ndarray. Starting part of the cumulative distribution function
        rhs: One dimensional ndarray. Right hand side of the histogram, used for calculation of the start_cdf if not provided
        kwargs: Can be ignored, used only for compatibility

        returns:
        Calculated feature value as described before.
        """
        try:
            start_band = kwargs['start_band']
            assert start_band is not None
        except KeyError:
            raise TypeError

        start_cdf = (np.cumsum(start_band, axis=1).T / np.sum(start_band, axis=1)).T
        uniform_cdf = np.linspace(0, 1, start_cdf.shape[1])

        unif_dist = abs((start_cdf - uniform_cdf)).sum(axis=1) / start_cdf.shape[1]

        return np.expand_dims(unif_dist, axis=1)

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f"{self.name}"]

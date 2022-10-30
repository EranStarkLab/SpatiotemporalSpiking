import numpy as np


class Jump(object):
    """
    This feature compares the middle band of the histogram to a linear change.
    """

    def __init__(self):
        self.name = 'Jump_index'

    @staticmethod
    def calculate_feature(**kwargs):
        """
        inputs:
        rhs: One dimensional ndarray. Right hand side of the histogram, used for calculation of the long-band if not provided
        kwargs: Can be ignored, used only for compatibility

        returns:
        Calculated feature value as described before.
        """
        try:
            mid_band = kwargs['mid_band']
            assert mid_band is not None
        except KeyError:
            raise TypeError

        mid_cdf = (np.cumsum(mid_band, axis=1).T / np.sum(mid_band, axis=1)).T
        uniform_cdf = np.linspace(0, 1, mid_cdf.shape[1])

        result = abs((mid_cdf - uniform_cdf)).sum(axis=1) / mid_cdf.shape[1]

        return np.expand_dims(result, axis=1)

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f'{self.name}']

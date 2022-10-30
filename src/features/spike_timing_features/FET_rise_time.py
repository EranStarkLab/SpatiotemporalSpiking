import numpy as np
import math


class RiseTime(object):
    """
    This feature estimates the firing pattern based on the starting band cumulative distribution function
    """

    def __init__(self):
        self.name = 'Rise_time'

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

        ach_rise_time = (start_cdf > 1 / math.e).argmax(axis=1)

        return np.expand_dims(ach_rise_time, axis=1)

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f'{self.name}']

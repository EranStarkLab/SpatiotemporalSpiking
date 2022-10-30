import numpy as np


class FiringRate(object):
    """
    This feature calculates the firing rate based on the ISIs
    """

    def __init__(self):
        self.name = 'Firing_rate'

    @staticmethod
    def calculate_feature(spike_train, chunks):
        """
        inputs:


        returns:
        Calculated feature value as described before.
        """
        isis = np.convolve(spike_train, [0.5, 0, -0.5], mode='same')
        end_spike = spike_train.size - 1
        chunk_isis = np.array([isis[chunk[(0 < chunk) * (chunk < end_spike)]].mean() for chunk in chunks])
        chunk_rates = 1 / chunk_isis

        return np.expand_dims(chunk_rates, axis=1)

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f'{self.name}']

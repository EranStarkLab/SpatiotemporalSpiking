import numpy as np


def calc_der(spike):
    """
    inputs:
    spike: the spike to be processed; it is an ndarray with TIMESTEPS * UPSAMPLE entries

    returns:
    The first order derivative of the spike calculated according to y'(t)=y(t+1)-y(t)
    """
    first_der = np.convolve(spike, [1, -1], mode='valid')

    return first_der


class MaxSpeed(object):
    """
    This feature times the duration after the depolarization in which the repolarization is at-least in the pace at the
    start*
    * start as defined by the class's field.
    """

    def __init__(self, start=131):
        # the index from which we shall start looking (remember that the spikes are alligned to have maximal
        # depolarization at ~128
        self.start = start

        self.name = 'Max_speed'

    def calculate_feature(self, spike_lst):
        """
        inputs:
        spike_lst: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        result = [self.calc_feature_spike(spike.get_data()) for spike in spike_lst]
        result = np.asarray(result)
        return result

    def calc_feature_spike(self, spike):
        """
        inputs:
        spike: the spike to be processed; it is an ndarray with TIMESTEPS * UPSAMPLE entries

        The function calculates the max speed feature as described above

        returns: a list containing the max speed value
        """
        der = calc_der(spike)
        der_roi = der[self.start:]
        ret = (der_roi > der_roi[0]).sum()

        return [ret]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f'{self.name}']

import numpy as np


def calc_second_der(spike):
    """
    inputs:
    spike: the spike to be processed; it is an ndarray with TIMESTEPS * UPSAMPLE entries

    returns:
    The second order derivative of the spike calculated according to y'(t)=y(t+1)-y(t)
    """
    first_der = np.convolve(spike, [1, -1], mode='valid')
    second_der = np.convolve(first_der, [1, -1], mode='valid')

    return second_der


class SmileCry(object):
    """
    This feature sums the second derivative in a window after the depolarization. This should allow us to asses the
    convexity of the spike in this section.
    """

    def __init__(self, start=170, end=250):
        # the start and end constants correspond to 0.26ms to 0.76ms after depolarization based on a sampling rate of
        # 20kHz and an upsampling by a factor of 8 (assuming depolarization is reached at the 128th timestep).
        self.start = start  # start of the region of interest in relation to the depolarization
        self.end = end  # end of the region of interest in relation to the depolarization

        self.name = 'Smile_cry'

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

        The function calculates the smile-cry feature as described above.

        returns: a list containing the smile-cry value
        """
        der = calc_second_der(spike)
        roi = der[self.start: self.end]

        ret = np.sum(roi)

        return [ret]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f'{self.name}']

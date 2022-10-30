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


class BreakMeasurement(object):
    """
    This feature aims at the quantification of the pre-depolarization "bump". Lower values generally indicate the
    existence of a "bump" while higher values indicate its absence.
    """

    def __init__(self, start=-48, end=-13, mul_const=1):
        # the start and end constants correspond to 0.3ms to 0.085ms before depolarization based on a sampling rate of
        # 20kHz and an upsampling by a factor of 8.
        self.start = start  # start of the region of interest in relation to the depolarization
        self.end = end  # end of the region of interest in relation to the depolarization
        # constants used for the calculation of the final value
        self.mul_const = mul_const

        self.name = 'Break_measure'

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

        The function calculates the break measurement as described above

        returns: a list containing the break measurement
        """
        dep_ind = np.argmin(spike)
        der = calc_second_der(spike)
        roi = der[dep_ind + self.start: dep_ind + self.end]

        ret = self.mul_const * np.sum(roi)

        return [ret]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f'{self.name}']

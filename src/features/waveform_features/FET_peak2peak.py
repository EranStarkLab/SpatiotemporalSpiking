import numpy as np


class Peak2Peak(object):
    """
    This feature evaluates the time and electrical difference between the lowest point (maximal depolarization) and
    highest point (maximal hyperpolarization).
    """

    def __init__(self):
        self.name = 'TTP'

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

    @staticmethod
    def calc_feature_spike(spike):
        """
        inputs:
        spike: the spike to be processed; it is an ndarray with TIMESTEPS * UPSAMPLE entries

        The function calculates the measurements as described above (also see return statement)

        returns: a list containing peak_2_peak (electrical difference between depolarization and hyperpolarization and
        through_2_peak (time difference between depolarization and hyperpolarization in terms of index difference)
        """
        # find timestamps for depolarization in ok channels, filter again to assure depolarization is reached before the
        # end
        dep_ind = np.argmin(spike)
        dep = spike[dep_ind]
        if dep_ind == len(spike) - 1:  # if max depolarization is reached at the end, it indicates noise
            return [0, 0]

        trun_spike = spike[dep_ind + 1:]
        hyp_ind = trun_spike.argmax() + dep_ind + 1
        hyp = spike[hyp_ind]

        peak_2_peak = hyp - dep
        trough_2_peak = hyp_ind - dep_ind

        return [peak_2_peak, trough_2_peak]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f'{self.name}_magnitude', f'{self.name}_duration']

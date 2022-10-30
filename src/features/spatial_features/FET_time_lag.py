import numpy as np

from constants import TIMESTEPS, UPSAMPLE

# There are two options of reduction for the da vector:
# ss - sum of squares
# sa - sum of absolutes


class TimeLagFeature(object):
    """
    This feature calculates the time difference between the main channel and all other channels in terms of
    maximal depolarization, and the following after hyperpolarization.
    The feature only takes into consideration channels that have crossed a certain threshold, determined by the
    maximal depolarization of the main channel.
    """

    def __init__(self, ratio=0.25, data_name='dep'):
        # Indicates the percentage of the maximum depolarization that will be considered as a threshold
        self.ratio = ratio

        self.name = 'Time-lag'
        self.data = data_name

    def set_data(self, new_data):
        self.data = new_data

    def calculate_feature(self, spike_lst, amps):
        """
        inputs:
        spike_lst: A list of Spike object that the feature will be calculated upon.

        returns:
        A matrix in which entry (i, j) refers to the j metric of Spike number i.
        """
        result = [self.calc_feature_spike(spike.get_data(), amp) for spike, amp in zip(spike_lst, amps)]
        result = np.asarray(result)
        return result

    def calc_feature_spike(self, spike, amp):
        """
        inputs:
        spike: the spike to be processed; it is a matrix with the dimensions of (NUM_CHANNELS, TIMESTEPS * UPSAMPLE)

        The function calculates different time lag features of the spike

        returns: a list containing the following values:
        -red: the reduction of the depolarization vector (i.e
        the vector that indicates the time difference of maximal depolarization between each channel and the main
        channel)
        -sd: the standard deviation of the depolarization vector
        """
        # remove channels with lower amplitude than required
        max_amp = np.max(amp)
        fix_inds = amp >= self.ratio * max_amp
        amp = amp[fix_inds]
        spike = spike[fix_inds]

        # find timestamps for the event in ok channels, filter again to assure the event is reached before the end
        ind = np.argmin(spike, axis=1)
        # if event is reached at the end, it indicates noise
        fix_inds = ind < (TIMESTEPS * UPSAMPLE - 1)
        amp = amp[fix_inds]
        ind = ind[fix_inds]
        spike = spike[fix_inds]
        if spike.shape[0] <= 1:  # if no channel passes filtering return zeros (or if only one channel)
            return [0, 0]

        # offset according to the main channel
        # set main channel to be the one with highest t2p
        main_chn = amp.argmax()
        rel = ind - ind[main_chn]  # offsetting

        # calculate sd of event time differences
        sd = np.std(rel)

        # calculate reduction
        red = np.mean(rel ** 2)

        return [red, sd]

    @property
    def headers(self):
        """
        Returns a list of titles of the different metrics
        """
        return [f"{self.data}_{self.name}_SS", f"{self.data}_{self.name}_SD"]

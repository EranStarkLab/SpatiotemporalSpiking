import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections.abc import Iterable
from constants import SAMPLE_RATE, NUM_CHANNELS, TIMESTEPS
from constants import PYR_COLOR, PV_COLOR, UT_COLOR, LIGHT_PYR, LIGHT_PV, LIGHT_UT


class Spike(object):
    def __init__(self, data=None):
        self.data = data  # Will contain data from 8 channels each with multiple samples over time (originally 32)

    def is_punit(self):
        """
        The function checks if a spike is of a positive-unit and returns the result
        """
        main_channel = self.data[np.argmax(np.absolute(self.data)) // self.data.shape[-1]]
        main_channel = main_channel[3:-3]  # in some cases the hyperpolarization at the edges was stronger than the
        # depolarization, causing wrong conclusion
        abs_diff = np.absolute(main_channel)
        arg_max = np.argmax(abs_diff, axis=0)  # the axis specification has no effect, just for clarification
        if main_channel[arg_max] > 0:
            return True
        return False

    def fix_punit(self):
        """
        The function flips the spike
        """
        self.data = self.data * -1

    def get_data(self):
        return self.data

class Cluster(object):
    def __init__(self, label=-1, filename=None, num_within_file=None, shank=None, spikes=None, timings=None):
        self.label = label
        self.filename = filename  # recording session
        self.num_within_file = num_within_file  # cluster ID
        self.shank = shank  # shank number
        self.spikes = spikes  # list of Spike
        self.np_spikes = None  # np array of spikes, used for optimization
        self.timings = timings

    def add_spike(self, spike, time):
        """
        The function receives a spike to append to the cluster (unit)
        """
        if self.spikes is not None:
            self.spikes.append(spike)
        else:
            self.spikes = [spike]

        if self.timings is not None:
            # we divide by the sample rate to get the time in seconds and multiply by 1000 to get the time in ms
            self.timings.append(1000 * time / SAMPLE_RATE)
        else:
            self.timings = [1000 * time / SAMPLE_RATE]

    def get_unique_name(self):
        """
        The function returns a unique name based on the cluster's fields
        """
        return self.filename + "_" + str(self.shank) + "_" + str(self.num_within_file)

    def calc_mean_waveform(self):
        """
        The function calculates the mean waveform (i.e. average spike of the cluster)
        """
        if self.np_spikes is None:  # for faster processing
            self.finalize_cluster()
        return Spike(data=self.np_spikes.mean(axis=0))

    def fix_punits(self):
        """
        The function checks if the cluster/unit is a positive-unit and flips it if so
        This is determined by the mean waveform
        """
        mean_spike = self.calc_mean_waveform()
        if mean_spike.is_punit():
            print(f"{self.get_unique_name()} was found to be a punit")
            self.np_spikes = self.np_spikes * -1
            return True
        return False

    def finalize_cluster(self):
        """
        The function transforms the spike list to a single numpy array, this is used for faster processing later on
        """
        shape = (len(self.spikes), NUM_CHANNELS, TIMESTEPS)
        self.np_spikes = np.empty(shape)
        for i, spike in enumerate(self.spikes):
            self.np_spikes[i] = spike.get_data()
        self.timings = np.array(self.timings)

    def save_cluster(self, path):
        if self.np_spikes is None:  # for faster processing
            self.finalize_cluster()
        if not os.path.isdir(path):
            os.mkdir(path)
        np.save(path + self.get_unique_name() + '__' + str(self.label) + '__' + 'spikes', self.np_spikes)
        np.save(path + self.get_unique_name() + '__' + str(self.label) + '__' + 'timings', self.timings)

    def load_cluster(self, path):
        """

        Parameters
        ----------
        path : String; path to cluster information. Each path contains partial information and two files are needed
        for each cluster

        Returns
        -------
        None. Cluster is updated in place.
        """
        path_elements = re.split("//|/", path)[-1].split('__')
        unique_name_elements = path_elements[0].split('_')
        self.filename = str.join('_', unique_name_elements[0: -2])
        self.shank = int(unique_name_elements[-2])
        self.num_within_file = int(unique_name_elements[-1])
        self.label = int(path_elements[-2])

        if 'spikes' in path_elements[-1]:
            self.spikes = np.load(path)
            self.np_spikes = np.load(path)
        elif 'timing' in path_elements[-1]:
            self.timings = np.load(path)

    def copy_cluster(self):
        """

        Returns
        -------
        New cluster with copy of the fields of the input cluster.
        """
        new_cluster = Cluster()
        new_cluster.label = self.label
        new_cluster.filename = self.filename  # recording session
        new_cluster.num_within_file = self.num_within_file  # cluster ID
        new_cluster.shank = self.shank  # shank number
        new_cluster.spikes = np.copy(self.np_spikes)
        new_cluster.np_spikes = np.copy(self.np_spikes)  # np array of spikes, used for optimization
        new_cluster.timings = np.copy(self.timings)

        return new_cluster

    def assert_legal(self):
        """

        Returns
        -------
        Trur iff all information of the cluster is present (i.e., both files were loaded)
        """
        if self.filename is None or \
                self.num_within_file is None or \
                self.shank is None or \
                self.spikes is None or \
                self.np_spikes is None or \
                self.timings is None:
            print(self.filename, self.num_within_file, self.shank)
            return False
        return True

    def plot_cluster(self, save=False, path=None, name=None):
        """

        Parameters
        ----------
        save : Bool; Whether to save or display the plot
        path : String; Path to save the plot in
        name : String; name identifier to save the plot

        Returns
        -------
        None. Plot is saved or displayed
        """
        fig, ax = plt.subplots(NUM_CHANNELS, 1, sharex=True, sharey=True, figsize=(3, 20))
        mean_spike = self.np_spikes.mean(axis=0)
        std_spike = self.np_spikes.std(axis=0)
        color1 = PYR_COLOR if self.label == 1 else PV_COLOR if self.label == 0 else UT_COLOR
        color2 = LIGHT_PYR if self.label == 1 else LIGHT_PV if self.label == 0 else LIGHT_UT
        for i, c_ax in enumerate(ax[::-1]):
            mean_channel = mean_spike[i]
            std_channel = std_spike[i]
            c_ax.plot(np.arange(TIMESTEPS), mean_channel, c=color1)
            c_ax.fill_between(np.arange(TIMESTEPS), mean_channel - std_channel, mean_channel + std_channel,
                              color=color2, alpha=0.2)
            if i != 0:
                c_ax.axis('off')
        if not save:
            fig.suptitle(f"Cluster {self.get_unique_name()} of type {'PYR' if self.label == 1 else 'IN' if self.label == 0 else 'UT' }")
            plt.show()
        if save:
            if name is None:
                name = ''
            plt.savefig(path + f"{name}_{self.get_unique_name()}.pdf", transparent=True)

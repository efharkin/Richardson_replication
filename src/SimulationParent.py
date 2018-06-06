#%% IMPORT MODULES

import abc

import numpy as np
from scipy import stats


#%% DEFINE SIMULATION PARENT CLASS

class Simulation(abc.ABC):

    @abc.abstractmethod
    def __init__(self, I, I_N, mod, replicates = 1, V0 = 0, dt = 0.1):

        pass

        """
        SAMPLE IMPLEMENTATION:

        self._mod = deepcopy(mod)

        I, V_mat, I_g1, spks, dt = self._mod.simulate(I, I_N, V0, replicates, dt)

        self.I      = I         # Injected current (nA)
        self.V      = V_mat     # Somatic voltage (mV)
        self.I_g1   = I_g1      # Current passed by conductance g1 (nA)
        self.spks   = spks      # Boolean vector of spks
        self.dt     = dt        # Simulation timestep
        """


    ### Methods to get various transformations of spiketrain
    def get_spk_mat(self):

        """
        Return a boolean spk matrix ([replicate, time] dimensionality).
        """

        return self.spks

    def get_spk_inds(self):

        """
        Return a list of arrays with spk inds for each replicate.
        """

        spk_inds_ls = []

        for i in range(self.replicates):
            spk_inds_ls.append(np.where(self.spks[i, :])[0])

        return spk_inds_ls

    def get_spk_times(self):

        """
        Return a list of arrays with spk times in ms.
        """

        spk_times_ls = self.get_spk_inds()

        for i in range(self.replicates):
            spk_times_ls[i] = spk_times_ls[i].astype(np.float64) * self.dt

        return spk_times_ls

    def get_firing_rate(self, bin_width = 10, return_I = False):

        """
        Compute binned firing rate

        Inputs:

            bin_width (float or None)
            --  Width of bin (ms) to use for firing rate calculation. Binned by timestep if set to None.

        Returns:
            Tuple of bin centres and binned firing rate.
        """

        spks_sum = self.spks.sum(axis = 0)

        if bin_width is None:

            inst_firing_rate = spks_sum / (self.replicates * self.dt) * 1e3

            if not return_I:
                return (self.get_t_vec(), inst_firing_rate)
            else:
                return (self.get_t_vec(), inst_firing_rate, self.I.mean(axis = 0))
        else:

            firing_rate_bins = np.arange(0, self.get_t_vec()[-1] + bin_width - self.dt, bin_width)
            binned_spks_sum, bin_edges, _ = stats.binned_statistic(
                self.get_t_vec(), spks_sum, 'sum', firing_rate_bins
                )
            binned_firing_rate = binned_spks_sum / (self.replicates * bin_width) * 1e3

            fr_bin_centres = (firing_rate_bins[1:] + firing_rate_bins[:-1]) / 2

            if not return_I:
                return (fr_bin_centres, binned_firing_rate)
            else:
                binned_I, _, _ = stats.binned_statistic(
                self.get_t_vec(), self.I.mean(axis = 0), 'mean', firing_rate_bins
                )

                return (fr_bin_centres, binned_firing_rate, binned_I)


    ### Method to get time vector
    def get_t_vec(self):

        """
        Return a time support vector for any one replicate.
        """

        return np.arange(0, self.I.shape[1] * self.dt, self.dt)

    def get_t_mat(self):

        """
        Return a time support matrix of the same shape as simulation.I, simulation.V, etc.
        """

        return np.tile(self.get_t_vec(), (self.I.shape[0], 1))


    ### Method to get replicates
    @abc.abstractmethod
    def replicates(self):
        pass

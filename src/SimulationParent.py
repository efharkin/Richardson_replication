#%% IMPORT MODULES

import abc

import numpy as np
import matplotlib.pyplot as plt
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

        return np.arange(0, self.I.shape[1] * self.dt, self.dt)[:self.I.shape[1]]

    def get_t_mat(self):

        """
        Return a time support matrix of the same shape as simulation.I, simulation.V, etc.
        """

        return np.tile(self.get_t_vec(), (self.I.shape[0], 1))


    ### Method to get replicates
    @abc.abstractmethod
    def replicates(self):
        pass


    ### Methods for gain analysis
    @staticmethod
    def _dft_single_freq(signal, f, fs, return_complex = True):

        """
        Compute the discrete fourier transform at a specific frequency.

        Inputs:

            signal (list-like)

            f (float)
            --  Frequency at which to compute the transform

            fs (float)
            --  Sampling frequency of the signal

            return_complex (bool; default True)
            --  Return complex result of the transform. Returns a tuple of amplitude and phase (in radians) if set to False.

        Returns:
            Result of the fourier transform as either a complex number or tuple of real-valued amplitude and phase, depending on the setting of `return_complex`.
        """

        exponential_term = np.exp(-1j * 2 * np.pi * np.arange(len(signal)) * f / fs)

        coefficient = np.sum(np.array(signal) * exponential_term)

        if return_complex:
            return coefficient
        else:
            amplitude   = np.absolute(coefficient)
            phase       = np.angle(coefficient)

            return (amplitude, phase)


    def extract_IO_gain_phase(self, f, subthreshold = False, bin_width = 1, discard_cycles = 2, plot = False):

        """
        Extract the gain and phase-shift of the firing response vs. input current at a specified frequency.

        Inputs:

            f (float)
            --  Frequency at which to get gain and phase shift.

            subthreshold (bool; default False)
            --  Calculate the gain/phase-shift on subthreshold voltage rather than firing rate.

            bin_width (float; default 1ms)
            --  Firing rate bin size (ms). (Only used if subthreshold set to False.)

            discard_cycles (int; default 2)
            --  Number of periods of f to discard from the beginning of the signal.

            plot (bool; default False)
            --  Make a diagnostic plot of signals being compared, along with calculated values of gain and phase-shift.

        Returns:
            Tuple of gain and phase-shift (radians).
        """

        if not subthreshold:
            # Get binned firing rate and current.
            t, output_signal, I_inj = self.get_firing_rate(bin_width, return_I = True)
        else:

            # Subthreshold gain cannot be computed if there are spks.
            # Check for spks and abort if any are found.
            if np.any(self.spks):
                raise ValueError('Subthreshold gain cannot be computed if there are spikes in the recording.')

            # Get subthreshold voltage and injected current.
            t = self.get_t_vec()
            output_signal = self.V.mean(axis = 0)
            I_inj = self.I.mean(axis = 0)

            bin_width = self.dt


        # Discard `discard_cycles` periods of f from beginning of all signals.
        # Avoids distortion from non-equilibrium state of simulation at initial condition.
        t               = t[int(discard_cycles * 1e3 / (f * bin_width)):]
        output_signal   = output_signal[int(discard_cycles * 1e3 / (f * bin_width)):]
        I_inj           = I_inj[int(discard_cycles * 1e3 / (f * bin_width)):]

        # Extract gain and phase-shift.
        I_ampli, I_phase = self._dft_single_freq(I_inj, f, 1e3 / bin_width, return_complex = False)
        output_ampli, output_phase = self._dft_single_freq(output_signal, f, 1e3 / bin_width, return_complex = False)

        gain = output_ampli / I_ampli
        phase_shift = output_phase - I_phase

        # Optionally, make a diagnostic plot.
        if plot:

            plt.figure(figsize = (8, 7))

            plt.suptitle('Gain plot at {}Hz'.format(f))

            output_plot = plt.subplot(111)
            output_plot.plot(t, output_signal, 'r-')
            output_plot.set_ylabel('Output', color = 'r')
            output_plot.set_xlabel('Time (ms)')

            I_plot = output_plot.twinx()
            I_plot.plot(t, I_inj, 'k-')
            I_plot.set_ylabel('I (nA)')

            output_plot.text(1, 1.02,
            'Gain: {:.3f}\nPhase-shift: {:.1f}'.format(gain, phase_shift),
            ha = 'right', va = 'bottom', transform = output_plot.transAxes)

            plt.subplots_adjust(top = 0.9)

            plt.show()

        # Return gain and phase-shift.
        return (gain, phase_shift)

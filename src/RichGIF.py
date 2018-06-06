#%% IMPORT MODULES

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numba as nb

import sys
sys.path.append('./src')

from SimulationParent import Simulation


#%% DEFINE GIF CLASS

class model(object):

    def __init__(self, C, g, g1, tau1, theta, reset):

        self.C = C          # Membrane capacitance
        self.g = g          # Leak conductance

        self.g1 = g1        # Extra conductance
        self.tau1 = tau1    # Time constant of extra conductance

        self.theta = theta  # Spike threshold
        self.reset = reset  # Post-spike voltage reset


    def simulate(self, I, I_N, V0, replicates = 1, dt = 0.1):

        """
        Simulate voltage.

        Returns:
            Tuple of I, V, I_g1, spks, and simulation dt.

            I, V, I_g1, and spks are matrices with dimensionality [replicate, time].
        """

        I = np.tile(I, (replicates, 1))

        I_rand = I_N * np.random.normal(size = I.shape)

        sim_tensor = self._simulate(I, I_rand, V0, self.C, self.g, self.g1, self.tau1, self.theta, self.reset, dt)

        V_mat = sim_tensor[0, :, :]
        I_g1 = sim_tensor[1, :, :]
        spks = sim_tensor[2, :, :].astype(np.bool)

        return (I, V_mat, I_g1, spks, dt)


    @staticmethod
    @nb.jit(nb.float64[:,:,:](
        nb.float64[:,:],
        nb.float64[:,:],
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64
        ))
    def _simulate(I, I_rand, V0, C, g, g1, tau1, theta, reset, dt):

        """
        Private numba-accelerated method for simulation.
        Called by GIF_mod.simulate().
        """

        # Create vectors to store output
        V_mat = np.empty(I.shape, dtype = np.float64)
        w_mat = np.empty(I.shape, dtype = np.float64)
        spks = np.zeros(I.shape, dtype = np.bool)

        # Set initial condition
        V_mat[:, 0] = V0
        w_mat[:, 0] = V_mat[:, 0]

        # Integrate over time
        t = 0
        while t < (I.shape[1] - 1):

            # Integrate conductance
            dw_t = (V_mat[:, t] - w_mat[:, t]) / tau1 * dt
            w_mat[:, t + 1] = w_mat[:, t] + dw_t

            # Detect spiking neurons
            spiking_neurons_t = V_mat[:, t] >= theta

            # Apply spiking rule to spiking neurons
            spks[spiking_neurons_t, t]          = True
            V_mat[spiking_neurons_t, t + 1]     = reset

            # Integrate voltage for all neurons,
            # but only apply to non-spiking cells.
            dV_t_deterministic = (-g * V_mat[:, t] - g1 * w_mat[:, t] + I[:, t]) / C * dt
            dV_t_stochastic = I_rand[:, t] / C * np.sqrt(dt)
            dV_t_total = dV_t_deterministic + dV_t_stochastic

            V_mat[~spiking_neurons_t, t + 1] = V_mat[~spiking_neurons_t, t] + dV_t_total[~spiking_neurons_t]

            # Increment t
            t += 1

        # Return output in a tensor
        return np.array([V_mat, g1 * w_mat, spks])


class simulation(Simulation):

    def __init__(self, I, I_N, mod, replicates = 1, V0 = 0, dt = 0.1):

        self._mod = deepcopy(mod) # Attach a copy of model just in case

        I, V_mat, I_g1, spks, dt = self._mod.simulate(I, I_N, V0, replicates, dt)

        self.I      = I         # Injected current (nA)
        self.V      = V_mat     # Somatic voltage (mV)
        self.I_g1   = I_g1      # Current passed by conductance g1 (nA)
        self.spks   = spks      # Boolean vector of spks
        self.dt     = dt        # Simulation timestep


    ### Method to get replicates
    @property
    def replicates(self):

        inferred_replicates = [self.I.shape[0],
                               self.V.shape[0],
                               self.I_g1.shape[0],
                               self.spks.shape[0]]

        assert all([inferred_replicates[0] == r for r in inferred_replicates]), 'Not all attrs have same no of replicates'

        return inferred_replicates[0]


    ### Plotting methods
    def basic_plot(self):

        plt.figure(figsize = (8, 6))

        spec = plt.GridSpec(4, 1, height_ratios = [0.2, 1, 1, 0.5], hspace = 0.4)

        t_mat_transpose = self.get_t_mat().T

        I_plot = plt.subplot(spec[3, :])
        plt.plot(t_mat_transpose, self.I.T, 'k-', alpha = 1/self.replicates)
        plt.ylabel('I (nA)')

        plt.subplot(spec[0, :], sharex = I_plot)
        plt.plot(t_mat_transpose, self.spks.T, 'k-', alpha = 1/self.replicates)
        plt.ylabel('Spks')
        plt.ylim(-0.05, 1.05)

        plt.subplot(spec[1, :], sharex = I_plot)
        plt.plot(t_mat_transpose, self.V.T, 'k-', alpha = 1/self.replicates)
        plt.ylabel('V (mV)')

        plt.subplot(spec[2, :], sharex = I_plot)
        plt.plot(t_mat_transpose, self.I_g1.T, 'k-', alpha = 1/self.replicates)
        plt.ylabel('I_g1 (nA)')

        plt.show()


    def firing_rate_plot(self, save_path = None, bin_width = 10):

        plt.figure(figsize = (8, 6))

        spec = plt.GridSpec(3, 1)

        t_mat_transpose = self.get_t_mat().T

        sample_neuron_plot = plt.subplot(spec[0, :])
        plt.title('A. Sample trace', loc = 'left')
        V_trace = self.V.T[:, 0] - 70
        V_trace[self.spks[0, :]] = 0
        plt.plot(t_mat_transpose[:, 0], V_trace, 'k-', linewidth = 0.5)
        sample_neuron_plot.set_xticklabels([])
        plt.ylabel('Voltage (mV)')

        raster_plot = plt.subplot(spec[1, :])
        plt.title('B. Raster plot', loc = 'left')
        spk_times = self.get_spk_times()
        for rep in range(self.replicates):
            plt.plot(spk_times[rep], [rep] * len(spk_times[rep]), '|', color = 'k')
        raster_plot.set_xticklabels([])
        plt.ylabel('Replicate')

        firing_rate_plot = plt.subplot(spec[2, :])
        plt.title('C. Mean firing rate', loc = 'left')
        t_binned, binned_firing_rate = self.get_firing_rate(bin_width = bin_width)
        plt.bar(t_binned, binned_firing_rate, width = bin_width, facecolor = 'none', edgecolor = 'k')
        plt.ylabel('Rate (Hz)')
        plt.xlabel('Time (ms)')

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi = 300)

        plt.show()

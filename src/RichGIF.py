#%% IMPORT MODULES

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import numba as nb


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
        spks = sim_tensor[2, :, :]

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


class simulation(object):

    def __init__(self, I, I_N, mod, replicates = 1, V0 = 0, dt = 0.1):

        self._mod = deepcopy(mod) # Attach a copy of model just in case

        I, V_mat, I_g1, spks, dt = self._mod.simulate(I, I_N, V0, replicates, dt)

        self.I      = I         # Injected current (nA)
        self.V      = V_mat     # Somatic voltage (mV)
        self.I_g1   = I_g1      # Current passed by conductance g1 (nA)
        self.spks   = spks      # Boolean vector of spks
        self.dt     = dt        # Simulation timestep


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


    ### Method to get time vector
    def get_t_vec(self):

        """
        Return a time support vector for any one replicate.
        """

        return np.arange(0, int(self.I.shape[1] * self.dt), self.dt)

    def get_t_mat(self):

        """
        Return a time support matrix of the same shape as simulation.I, simulation.V, etc.
        """

        return np.tile(self.get_t_vec(), (self.I.shape[0], 1))


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

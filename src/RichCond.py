#%% IMPORT MODULES

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


#%% DEFINE NEURON MODEL CLASS

class model1(object):

    def __init__(self, C = 0.37, gl = 0.037, El = -68, gNa = 19.24, ENa = 55,
        tau_m = 0, gK = 7.4, EK = -90, gIh = 0.03, EIh = -41, tau_Ihf = 38,
        tau_Ihs = 319):

        self.C      = C             # Membrane capacitance (nF)
        self.gl     = gl            # Leak conductance (uS)
        self.El     = El            # Leak reversal (mV)

        self.gNa    = gNa           # Sodium conductance (uS)
        self.ENa    = ENa           # Sodium reversal (mV)
        self.tau_m  = tau_m         # Sodium activation tau (ms)

        self.gK     = gK            # Potassium conductance (uS)
        self.EK     = EK            # Potassium reversal (mV)

        self.gIh        = gIh       # Ih conductance (uS)
        self.EIh        = EIh       # Ih reversal (mV)
        self.tau_Ihf    = tau_Ihf   # Ih fast tau (ms)
        self.tau_Ihs    = tau_Ihs   # Ih slow tau (ms)


    def simulate(self, I, I_N, V0, replicates = 1, spk_detect_thresh = -30,
        spk_detect_tref = 2, dt = 0.1):

        """
        Simulate voltage.

        Inputs:

            spk_detect_thresh (numeric; default -30)
            --  Voltage threshold (mV) for online spike detection

            spk_detect_tref (numeric; default 2)
            --  'Absolute refractory period' (ms) used to prevent detecting the same spk multiple times.

        Returns:
            Tuple of I, V, m, h, n, Ihf, Ihs, spks, and dt

            I through spks are matrices with dimensionality [replicate, time].
        """

        I = np.tile(I, (replicates, 1))

        I_rand = I_N * np.random.normal(size = I.shape)

        sim_tensor = self._simulate(
            I, I_rand, V0, self.C, self.gl, self.El, self.gNa, self.ENa,
            self.gK, self.EK, self.gIh, self.EIh, self.tau_Ihf, self.tau_Ihs,
            spk_detect_thresh, spk_detect_tref, dt
        )

        V_mat       = sim_tensor[0, :, :]
        m_mat       = sim_tensor[1, :, :]
        h_mat       = sim_tensor[2, :, :]
        n_mat       = sim_tensor[3, :, :]
        Ihf_mat     = sim_tensor[4, :, :]
        Ihs_mat     = sim_tensor[5, :, :]
        spks_mat    = sim_tensor[6, :, :].astype(np.bool)

        return (I, V_mat, m_mat, h_mat, n_mat, Ihf_mat, Ihs_mat, spks_mat, dt)


    @staticmethod
    def _simulate(I, I_rand, V0, C, gl, El, gNa, ENa, gK, EK, gIh, EIh,
        tau_Ihf, tau_Ihs, spk_detect_thresh, spk_detect_tref, dt):

        """
        Private method for simulation.
        Called by GIF_mod.simulate().

        Ripe for acceleration with numba.jit(), except that numba throws an error when _simulate is called.
        """

        ### Define functions.
        integrate_gate = lambda x_inf_, x_0, tau_x_, dt_: (x_inf_ - x_0) / tau_x_ * dt_

        # Define gating functions for m, h, n
        x_inf = lambda alpha, beta, V: alpha(V) / (alpha(V) + beta(V))
        tau_x = lambda alpha, beta, V: 1 / (26.12 * (alpha(V) + beta(V)))

        alpha_m = lambda V: -0.1 * (V + 32) / (np.exp(-0.1 * (V + 32)) - 1)
        beta_m = lambda V: 4 * np.exp(-(V + 57)/18)

        alpha_h = lambda V: 0.07 * np.exp(-(V + 46)/20)
        beta_h = lambda V: 1 / (np.exp(-0.1 * (V + 16)) + 1)

        alpha_n = lambda V: -0.01 * (V + 36) / (np.exp(-0.1 * (V + 36)) - 1)
        beta_n = lambda V: 0.125 * np.exp(-(V + 46)/80)

        m_inf = lambda V: x_inf(alpha_m, beta_m, V)
        h_inf = lambda V: x_inf(alpha_h, beta_h, V)
        n_inf = lambda V: x_inf(alpha_n, beta_n, V)

        tau_h = lambda V: tau_x(alpha_h, beta_h, V)
        tau_n = lambda V: tau_x(alpha_n, beta_n, V)

        # Define gating functions for Ih
        Ihf_inf = lambda V: 1 / (1 + np.exp((V + 78) / 7))
        Ihs_inf = lambda V: 1 / (1 + np.exp((V + 78) / 7))


        ### Create matrices to store output
        V_mat = np.empty(I.shape, dtype = np.float64)
        m_mat = np.empty(I.shape, dtype = np.float64)
        h_mat = np.empty(I.shape, dtype = np.float64)
        n_mat = np.empty(I.shape, dtype = np.float64)
        Ihf_mat = np.empty(I.shape, dtype = np.float64)
        Ihs_mat = np.empty(I.shape, dtype = np.float64)
        spks_mat = np.zeros(I.shape, dtype = np.bool)


        ### Set initial conditions
        V_mat[:, 0]     = V0
        m_mat[:, 0]     = m_inf(V0)
        h_mat[:, 0]     = h_inf(V0)
        n_mat[:, 0]     = n_inf(V0)
        Ihf_mat[:, 0]   = Ihf_inf(V0)
        Ihs_mat[:, 0]   = Ihs_inf(V0)

        spk_detect_tref_ind = int(spk_detect_tref / dt)


        ### Integrate over time
        t = 0
        while t < (I.shape[1] - 1):

            V_t = V_mat[:, t]

            # Integrate gates
            m_mat[:, t + 1] = m_inf(V_t)
            h_mat[:, t + 1] = h_mat[:, t] + integrate_gate(h_inf(V_t), h_mat[:, t], tau_h(V_t), dt)
            n_mat[:, t + 1] = n_mat[:, t] + integrate_gate(n_inf(V_t), n_mat[:, t], tau_n(V_t), dt)
            Ihf_mat[:, t + 1] = Ihf_mat[:, t] + integrate_gate(Ihf_inf(V_t), Ihf_mat[:, t], tau_Ihf, dt)
            Ihs_mat[:, t + 1] = Ihs_mat[:, t] + integrate_gate(Ihs_inf(V_t), Ihs_mat[:, t], tau_Ihs, dt)

            # Integrate V
            I_conductances = (
                -gl * (V_t - El)
                - gNa * m_mat[:, t]**3 * h_mat[:, t] * (V_t - ENa)
                - gK * n_mat[:, t]**4 * (V_t - EK)
                - gIh * (0.8 * Ihf_mat[:, t] + 0.2 * Ihs_mat[:, t]) * (V_t - EIh)
            )

            dV_t_deterministic = (I_conductances + I[:, t]) / C * dt
            dV_t_stochastic = I_rand[:, t] / C * np.sqrt(dt)

            V_mat[:, t + 1] = V_t + dV_t_deterministic + dV_t_stochastic

            # Flag spks
            spks_t = np.logical_and(
                np.logical_and(V_t > spk_detect_thresh, dV_t_deterministic > 0),
                ~np.any(spks_mat[:, t-spk_detect_tref_ind:t])
            )
            spks_mat[spks_t, t] = True

            # Increment t
            t += 1


        ### Return output in a tensor
        return np.array([V_mat, m_mat, h_mat, n_mat, Ihf_mat, Ihs_mat, spks_mat])


#%% DEFINE SIMULATION CLASS

class simulation_m1(object):

    def __init__(self, I, I_N, mod, replicates = 1, V0 = 0, dt = 0.1):

        self._mod = deepcopy(mod) # Attach a copy of model just in case

        I, V_mat, m_mat, h_mat, n_mat, Ihf_mat, Ihs_mat, spks_mat, dt = (
            self._mod.simulate(I, I_N, V0, replicates, dt = dt)
        )

        self.I      = I         # Injected current (nA)
        self.V      = V_mat     # Somatic voltage (mV)
        self.m      = m_mat
        self.h      = h_mat
        self.n      = n_mat
        self.Ihf    = Ihf_mat
        self.Ihs    = Ihs_mat
        self.spks   = spks_mat  # Boolean vector of spks
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

    def get_firing_rate(self, bin_width = 10):

        """
        Compute binned firing rate

        Inputs:

            bin_width (float or None)
            --  Width of bin (ms) to use for firing rate calculation. Binned by timestep if set to None.

        Returns:
            Tuple of bin centres and binned firing rate.
        """

        inst_firing_rate = self.spks.sum(axis = 0) / (self.replicates * self.dt) * 1e3

        if bin_width is None:
            return (self.get_t_vec(), inst_firing_rate)
        else:
            firing_rate_bins = np.arange(0, self.get_t_vec()[-1] + bin_width - self.dt, bin_width)
            binned_firing_rate, bin_edges, _ = stats.binned_statistic(
                self.get_t_vec(), inst_firing_rate, 'mean', firing_rate_bins
                )

            fr_bin_centres = (firing_rate_bins[1:] + firing_rate_bins[:-1]) / 2

            return (fr_bin_centres, binned_firing_rate)


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
                               self.m.shape[0],
                               self.h.shape[0],
                               self.n.shape[0],
                               self.Ihf.shape[0],
                               self.Ihs.shape[0],
                               self.spks.shape[0]]

        assert all([inferred_replicates[0] == r for r in inferred_replicates]), 'Not all attrs have same no of replicates'

        return inferred_replicates[0]


    ### Plotting methods
    def basic_plot(self):

        plt.figure(figsize = (8, 6))

        spec = plt.GridSpec(3, 1, height_ratios = [0.2, 1, 0.5], hspace = 0.4)

        t_mat_transpose = self.get_t_mat().T

        I_plot = plt.subplot(spec[2, :])
        plt.plot(t_mat_transpose, self.I.T, 'k-', alpha = 1/self.replicates)
        plt.ylabel('I (nA)')

        plt.subplot(spec[0, :], sharex = I_plot)
        plt.plot(t_mat_transpose, self.spks.T, 'k-', alpha = 1/self.replicates)
        plt.ylabel('Spks')
        plt.ylim(-0.05, 1.05)

        plt.subplot(spec[1, :], sharex = I_plot)
        plt.plot(t_mat_transpose, self.V.T, 'k-', alpha = 1/self.replicates)
        plt.ylabel('V (mV)')

        plt.show()


    def firing_rate_plot(self, save_path = None, bin_width = 10):

        plt.figure(figsize = (8, 6))

        spec = plt.GridSpec(3, 1)

        t_mat_transpose = self.get_t_mat().T

        sample_neuron_plot = plt.subplot(spec[0, :])
        plt.title('A. Sample trace', loc = 'left')
        V_trace = self.V.T[:, 0]
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

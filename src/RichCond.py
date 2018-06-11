#%% IMPORT MODULES

from copy import deepcopy
import gc

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import numba as nb

import sys
sys.path.append('./src')

from SimulationParent import Simulation


#%% DEFINE NEURON MODEL CLASSES

class model1(object):

    def __init__(self, C = 0.37, gl = 0.037, El = -68, gNa = 19.24, ENa = 55,
        tau_m = 0, gK = 7.4, EK = -90, gIh = 0.03, EIh = -41, tau_Ihf = 38,
        tau_Ihs = 319):

        self._model_type = '1'

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


    def simulate(self, I, V0, replicates = 1, I_N = None, ge = None, Ee = None,
        gi = None, Ei = None, spk_detect_thresh = -30,
        spk_detect_tref = 2, dt = 0.1):

        """
        Simulate voltage.

        Inputs:

            ge (matrix or None; default None)
            --  Matrix of excitatory synaptic conductances of same shape as I. (gi for inhibitory)

            Ee (float or None; default None)
            --  Reversal potential of excitatory synaptic input

            spk_detect_thresh (numeric; default -30)
            --  Voltage threshold (mV) for online spike detection

            spk_detect_tref (numeric; default 2)
            --  'Absolute refractory period' (ms) used to prevent detecting the same spk multiple times.

        Returns:
            Tuple of I, V, m, h, n, Ihf, Ihs, spks, and dt

            I through spks are matrices with dimensionality [replicate, time].
        """

        I = np.tile(I, (replicates, 1))

        # Check for correct input.
        syn_are_None = np.array([x is None for x in [ge, Ee, gi, Ei]])

        if I_N is not None:
            if not all(syn_are_None):
                raise ValueError('Either I_N or ge/Ee/gi/Ei can be specified, but not both.')
        if any(~syn_are_None):
            if not all(~syn_are_None):
                raise ValueError('All of ge/Ee/gi/Ei should be specified, or all should be set to None.')

        # Select appropriate internal simulation method.
        if I_N is not None:
            I_rand = I_N * np.random.normal(size = I.shape)

            sim_tensor = self._simulate_N(
                I, V0, I_rand, self.C, self.gl, self.El, self.gNa, self.ENa,
                self.gK, self.EK, self.gIh, self.EIh, self.tau_Ihf, self.tau_Ihs,
                spk_detect_thresh, spk_detect_tref, dt
            )

        elif all(~syn_are_None):

            sim_tensor = self._simulate_syn(
                I, V0, ge, Ee, gi, Ei, self.C, self.gl, self.El, self.gNa,
                self.ENa, self.gK, self.EK, self.gIh, self.EIh, self.tau_Ihf,
                self.tau_Ihs, spk_detect_thresh, spk_detect_tref, dt
            )

        else:

            raise NotImplementedError('Purely deterministic simulation not implemented. Use I_N = 0 instead.')


        V_mat       = sim_tensor[0]
        spks_mat    = sim_tensor[1].astype(np.bool)

        return (I, V_mat, spks_mat, dt)


    @staticmethod
    def _simulate_N(I, V0, I_rand, C, gl, El, gNa, ENa, gK, EK, gIh, EIh,
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
        m_vec = np.empty(I.shape[0], dtype = np.float64)
        h_vec = np.empty(I.shape[0], dtype = np.float64)
        n_vec = np.empty(I.shape[0], dtype = np.float64)
        Ihf_vec = np.empty(I.shape[0], dtype = np.float64)
        Ihs_vec = np.empty(I.shape[0], dtype = np.float64)
        spks_mat = np.zeros(I.shape, dtype = np.bool)


        ### Set initial conditions
        V_mat[:, 0]     = V0
        m_vec[:]     = m_inf(V0)
        h_vec[:]     = h_inf(V0)
        n_vec[:]     = n_inf(V0)
        Ihf_vec[:]   = Ihf_inf(V0)
        Ihs_vec[:]   = Ihs_inf(V0)

        spk_detect_tref_ind = int(spk_detect_tref / dt)


        ### Integrate over time
        t = 0
        while t < (I.shape[1] - 1):

            V_t = V_mat[:, t]

            # Integrate V
            I_conductances = (
                -gl * (V_t - El)
                - gNa * m_vec**3 * h_vec * (V_t - ENa)
                - gK * n_vec**4 * (V_t - EK)
                - gIh * (0.8 * Ihf_vec + 0.2 * Ihs_vec) * (V_t - EIh)
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

            # Integrate gates
            m_vec = m_inf(V_t)
            h_vec += integrate_gate(h_inf(V_t), h_vec, tau_h(V_t), dt)
            n_vec += integrate_gate(n_inf(V_t), n_vec, tau_n(V_t), dt)
            Ihf_vec += integrate_gate(Ihf_inf(V_t), Ihf_vec, tau_Ihf, dt)
            Ihs_vec += integrate_gate(Ihs_inf(V_t), Ihs_vec, tau_Ihs, dt)

            del dV_t_deterministic, dV_t_stochastic, spks_t, V_t

            # Increment t
            t += 1


        ### Return output in a tensor
        return (V_mat, spks_mat)


    @staticmethod
    def _simulate_syn(I, V0, ge, Ee, gi, Ei, C, gl, El, gNa, ENa, gK, EK, gIh, EIh,
        tau_Ihf, tau_Ihs, spk_detect_thresh, spk_detect_tref, dt):

        """
        Private method for simulation.
        Called by GIF_mod.simulate().

        Ripe for acceleration with numba.jit(), except that numba throws an error when _simulate is called.
        """

        ### Define numba-vectorized functions
        @nb.vectorize([nb.float64(nb.float64)])
        def m_inf(V):
            alpha_m = -0.1 * (V + 32) / (np.exp(-0.1 * (V + 32)) - 1)
            beta_m = 4 * np.exp(-(V + 57)/18)
            return alpha_m / (alpha_m + beta_m)

        @nb.vectorize([nb.float64(nb.float64)])
        def h_inf(V):
            alpha_h = 0.07 * np.exp(-(V + 46)/20)
            beta_h = 1 / (np.exp(-0.1 * (V + 16)) + 1)
            return alpha_h / (alpha_h + beta_h)

        @nb.vectorize([nb.float64(nb.float64, nb.float64)])
        def d_h(V, h):

            alpha_h = 0.07 * np.exp(-(V + 46)/20)
            beta_h = 1 / (np.exp(-0.1 * (V + 16)) + 1)

            h_inf = alpha_h / (alpha_h + beta_h)
            tau_h = 1 / (26.12 * (alpha_h + beta_h))

            return (h_inf - h) / tau_h

        @nb.vectorize([nb.float64(nb.float64)])
        def n_inf(V):
            alpha_n = -0.01 * (V + 36) / (np.exp(-0.1 * (V + 36)) - 1)
            beta_n = 0.125 * np.exp(-(V + 46)/80)
            return alpha_n / (alpha_n + beta_n)

        @nb.vectorize([nb.float64(nb.float64, nb.float64)])
        def d_n(V, n):

            alpha_n = -0.01 * (V + 36) / (np.exp(-0.1 * (V + 36)) - 1)
            beta_n = 0.125 * np.exp(-(V + 46)/80)

            n_inf = alpha_n / (alpha_n + beta_n)
            tau_n = 1 / (26.12 * (alpha_n + beta_n))

            return (n_inf - n) / tau_n

        @nb.vectorize([nb.float64(nb.float64)])
        def Ihf_inf(V):
            return 1 / (1 + np.exp((V + 78) / 7))

        @nb.vectorize([nb.float64(nb.float64)])
        def Ihs_inf(V):
            return 1 / (1 + np.exp((V + 78) / 7))


        @nb.jit(nb.int8[:](nb.float64[:], nb.float64, nb.float64[:], nb.int8[:,:]), cache = True)
        def detect_spks(V_t, spk_detect_thresh, dV_t_deterministic, spks_subset):
            output = np.zeros(V_t.shape, dtype = np.int8)

            for i in range(len(V_t)):
                bool_flag = (V_t[i] > spk_detect_thresh) and (dV_t_deterministic[i] > 0) and (spks_subset[i, :].sum() == 0)
                if bool_flag:
                    output[i] = 1
                else:
                    continue

            return output



        ### Create matrices to store output
        V_mat = np.empty(I.shape, dtype = np.float64)
        m_vec = np.empty(I.shape[0], dtype = np.float64)
        h_vec = np.empty(I.shape[0], dtype = np.float64)
        n_vec = np.empty(I.shape[0], dtype = np.float64)
        Ihf_vec = np.empty(I.shape[0], dtype = np.float64)
        Ihs_vec = np.empty(I.shape[0], dtype = np.float64)
        spks_mat = np.zeros(I.shape, dtype = np.int8)


        ### Set initial conditions
        V_mat[:, 0]     = V0
        m_vec[:]     = m_inf(V0)
        h_vec[:]     = h_inf(V0)
        n_vec[:]     = n_inf(V0)
        Ihf_vec[:]   = Ihf_inf(V0)
        Ihs_vec[:]   = Ihs_inf(V0)

        spk_detect_tref_ind = int(spk_detect_tref / dt)


        ### Integrate over time
        t = 0
        while t < (I.shape[1] - 1):

            V_t = V_mat[:, t]

            # Integrate V
            I_conductances = (
                -gl * (V_t - El)
                - gNa * m_vec**3 * h_vec * (V_t - ENa)
                - gK * n_vec**4 * (V_t - EK)
                - gIh * (0.8 * Ihf_vec + 0.2 * Ihs_vec) * (V_t - EIh)
            )

            dV_t_deterministic = (I_conductances + I[:, t]) / C * dt
            dV_t_stochastic = (ge[:, t] * (V_t - Ee) + gi[:, t] * (V_t - Ei)) / C * dt

            V_mat[:, t + 1] = V_t + dV_t_deterministic + dV_t_stochastic

            # Flag spks
            spks_mat[:, t] = detect_spks(
                V_t, spk_detect_thresh, dV_t_deterministic,
                spks_mat[:, t-spk_detect_tref_ind:t]
            )

            # Integrate gates
            m_vec = m_inf(V_t)
            h_vec += d_h(V_t, h_vec) * dt
            n_vec += d_n(V_t, n_vec) * dt
            Ihf_vec += (Ihf_inf(V_t) - Ihf_vec) / tau_Ihf * dt
            Ihs_vec += (Ihs_inf(V_t) - Ihs_vec) / tau_Ihs * dt

            del dV_t_deterministic, dV_t_stochastic, V_t

            if t % 100 == 0:
                gc.collect()

            # Increment t
            t += 1


        ### Return output in a tensor
        return V_mat, spks_mat


class model2(object):

    def __init__(self, C = 0.37, gl = 0.037, El = -68, gNa = 19.24, ENa = 55,
        tau_m = 0, gK = 7.4, EK = -90, gNaP = 0.037, gKs = 2.59, tau_q = 6):

        self._model_type = '2'

        self.C      = C             # Membrane capacitance (nF)
        self.gl     = gl            # Leak conductance (uS)
        self.El     = El            # Leak reversal (mV)

        self.gNa    = gNa           # Sodium conductance (uS)
        self.ENa    = ENa           # Sodium reversal (mV)
        self.tau_m  = tau_m         # Sodium activation tau (ms)

        self.gK     = gK            # Potassium conductance (uS)
        self.EK     = EK            # Potassium reversal (mV)

        self.gNaP   = gNaP          # Persistent sodium maximal conductance (uS)

        self.gKs    = gKs           # Slow potassium maximal conductance (uS)
        self.tau_q  = tau_q         # Slow potassium tau (ms)


    def simulate(self, I, V0, replicates = 1, I_N = None, ge = None, Ee = None,
        gi = None, Ei = None, spk_detect_thresh = -30, spk_detect_tref = 2,
        dt = 0.1):

        """
        Simulate voltage.

        Inputs:

            spk_detect_thresh (numeric; default -30)
            --  Voltage threshold (mV) for online spike detection

            spk_detect_tref (numeric; default 2)
            --  'Absolute refractory period' (ms) used to prevent detecting the same spk multiple times.

        Returns:
            Tuple of I, V, m, h, n, p, q, spks, and dt

            I through spks are matrices with dimensionality [replicate, time].
        """

        I = np.tile(I, (replicates, 1))

        # Check for correct input.
        syn_are_None = np.array([x is None for x in [ge, Ee, gi, Ei]])

        if I_N is not None:
            if not all(syn_are_None):
                raise ValueError('Either I_N or ge/Ee/gi/Ei can be specified, but not both.')
        if any(~syn_are_None):
            if not all(~syn_are_None):
                raise ValueError('All of ge/Ee/gi/Ei should be specified, or all should be set to None.')

        if I_N is not None:

            I_rand = I_N * np.random.normal(size = I.shape)

            sim_tensor = self._simulate_N(
                I, V0, I_rand, self.C, self.gl, self.El, self.gNa, self.ENa,
                self.gK, self.EK, self.gNaP, self.gKs, self.tau_q,
                spk_detect_thresh, spk_detect_tref, dt
            )

        elif all(~syn_are_None):

            sim_tensor = self._simulate_syn(
                I, V0, ge, Ee, gi, Ei, self.C, self.gl, self.El, self.gNa, self.ENa,
                self.gK, self.EK, self.gNaP, self.gKs, self.tau_q,
                spk_detect_thresh, spk_detect_tref, dt
            )

        else:

            raise NotImplementedError('Purely deterministic simulation not implemented. Use I_N = 0 instead.')

        V_mat       = sim_tensor[0]
        spks_mat    = sim_tensor[1].astype(np.bool)

        return (I, V_mat, spks_mat, dt)


    @staticmethod
    def _simulate_N(I, V0, I_rand, C, gl, El, gNa, ENa, gK, EK, gNaP, gKs, tau_q,
        spk_detect_thresh, spk_detect_tref, dt):

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

        # Define gating functions for additional conductances
        p_inf = lambda V: 1 / (1 + np.exp(-(V + 51) / 5))
        q_inf = lambda V: 1 / (1 + np.exp(-(V + 34) / 6.5))


        ### Create matrices to store output
        V_mat = np.empty(I.shape, dtype = np.float64)
        m_vec = np.empty(I.shape[0], dtype = np.float64)
        h_vec = np.empty(I.shape[0], dtype = np.float64)
        n_vec = np.empty(I.shape[0], dtype = np.float64)
        p_vec = np.empty(I.shape[0], dtype = np.float64)
        q_vec = np.empty(I.shape[0], dtype = np.float64)
        spks_mat = np.zeros(I.shape, dtype = np.bool)


        ### Set initial conditions
        V_mat[:, 0]     = V0
        m_vec[:]     = m_inf(V0)
        h_vec[:]     = h_inf(V0)
        n_vec[:]     = n_inf(V0)
        p_vec[:]     = p_inf(V0)
        q_vec[:]     = q_inf(V0)

        spk_detect_tref_ind = int(spk_detect_tref / dt)


        ### Integrate over time
        t = 0
        while t < (I.shape[1] - 1):

            V_t = V_mat[:, t]

            # Integrate V
            I_conductances = (
                -gl * (V_t - El)
                - gNa * m_vec**3 * h_vec * (V_t - ENa)
                - gK * n_vec**4 * (V_t - EK)
                - gNaP * p_vec * (V_t - ENa)
                - gKs * q_vec * (V_t - EK)
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

            # Integrate gates
            m_vec = m_inf(V_t)
            h_vec += integrate_gate(h_inf(V_t), h_vec, tau_h(V_t), dt)
            n_vec += integrate_gate(n_inf(V_t), n_vec, tau_n(V_t), dt)
            p_vec = p_inf(V_t)
            q_vec += integrate_gate(q_inf(V_t), q_vec, tau_q, dt)

            del dV_t_deterministic, dV_t_stochastic, spks_t, V_t

            # Increment t
            t += 1


        ### Return output in a tensor
        return (V_mat, spks_mat)

    @staticmethod
    def _simulate_syn(I, V0, ge, Ee, gi, Ei, C, gl, El, gNa, ENa, gK, EK, gNaP, gKs, tau_q,
        spk_detect_thresh, spk_detect_tref, dt):

        """
        Private method for simulation.
        Called by GIF_mod.simulate().

        Ripe for acceleration with numba.jit(), except that numba throws an error when _simulate is called.
        """

        ### Define numba-vectorized functions
        @nb.vectorize([nb.float64(nb.float64)])
        def m_inf(V):
            alpha_m = -0.1 * (V + 32) / (np.exp(-0.1 * (V + 32)) - 1)
            beta_m = 4 * np.exp(-(V + 57)/18)
            return alpha_m / (alpha_m + beta_m)

        @nb.vectorize([nb.float64(nb.float64)])
        def h_inf(V):
            alpha_h = 0.07 * np.exp(-(V + 46)/20)
            beta_h = 1 / (np.exp(-0.1 * (V + 16)) + 1)
            return alpha_h / (alpha_h + beta_h)

        @nb.vectorize([nb.float64(nb.float64, nb.float64)])
        def d_h(V, h):

            alpha_h = 0.07 * np.exp(-(V + 46)/20)
            beta_h = 1 / (np.exp(-0.1 * (V + 16)) + 1)

            h_inf = alpha_h / (alpha_h + beta_h)
            tau_h = 1 / (26.12 * (alpha_h + beta_h))

            return (h_inf - h) / tau_h

        @nb.vectorize([nb.float64(nb.float64)])
        def n_inf(V):
            alpha_n = -0.01 * (V + 36) / (np.exp(-0.1 * (V + 36)) - 1)
            beta_n = 0.125 * np.exp(-(V + 46)/80)
            return alpha_n / (alpha_n + beta_n)

        @nb.vectorize([nb.float64(nb.float64, nb.float64)])
        def d_n(V, n):

            alpha_n = -0.01 * (V + 36) / (np.exp(-0.1 * (V + 36)) - 1)
            beta_n = 0.125 * np.exp(-(V + 46)/80)

            n_inf = alpha_n / (alpha_n + beta_n)
            tau_n = 1 / (26.12 * (alpha_n + beta_n))

            return (n_inf - n) / tau_n

        @nb.vectorize([nb.float64(nb.float64)])
        def p_inf(V):
            return 1 / (1 + np.exp(-(V + 51) / 5))

        @nb.vectorize([nb.float64(nb.float64)])
        def q_inf(V):
            return 1 / (1 + np.exp(-(V + 34) / 6.5))


        @nb.jit(nb.int8[:](nb.float64[:], nb.float64, nb.float64[:], nb.int8[:,:]), cache = True)
        def detect_spks(V_t, spk_detect_thresh, dV_t_deterministic, spks_subset):
            output = np.zeros(V_t.shape, dtype = np.int8)

            for i in range(len(V_t)):
                bool_flag = (V_t[i] > spk_detect_thresh) and (dV_t_deterministic[i] > 0) and (spks_subset[i, :].sum() == 0)
                if bool_flag:
                    output[i] = 1
                else:
                    continue

            return output


        ### Create matrices to store output
        V_mat = np.empty(I.shape, dtype = np.float64)
        m_vec = np.empty(I.shape[0], dtype = np.float64)
        h_vec = np.empty(I.shape[0], dtype = np.float64)
        n_vec = np.empty(I.shape[0], dtype = np.float64)
        p_vec = np.empty(I.shape[0], dtype = np.float64)
        q_vec = np.empty(I.shape[0], dtype = np.float64)
        spks_mat = np.zeros(I.shape, dtype = np.int8)


        ### Set initial conditions
        V_mat[:, 0]     = V0
        m_vec[:]     = m_inf(V0)
        h_vec[:]     = h_inf(V0)
        n_vec[:]     = n_inf(V0)
        p_vec[:]     = p_inf(V0)
        q_vec[:]     = q_inf(V0)

        spk_detect_tref_ind = int(spk_detect_tref / dt)


        ### Integrate over time
        t = 0
        while t < (I.shape[1] - 1):

            V_t = V_mat[:, t]

            # Integrate V
            I_conductances = (
                -gl * (V_t - El)
                - gNa * m_vec**3 * h_vec * (V_t - ENa)
                - gK * n_vec**4 * (V_t - EK)
                - gNaP * p_vec * (V_t - ENa)
                - gKs * q_vec * (V_t - EK)
            )

            dV_t_deterministic = (I_conductances + I[:, t]) / C * dt
            dV_t_stochastic = (ge[:, t] * (V_t - Ee) + gi[:, t] * (V_t - Ei)) / C * dt

            V_mat[:, t + 1] = V_t + dV_t_deterministic + dV_t_stochastic

            # Flag spks
            spks_mat[:, t] = detect_spks(V_t, spk_detect_thresh, dV_t_deterministic, spks_mat[:, (t-spk_detect_tref_ind):t])

            # Integrate gates
            m_vec = m_inf(V_t)
            h_vec += d_h(V_t, h_vec) * dt
            n_vec += d_n(V_t, n_vec) * dt
            p_vec = p_inf(V_t)
            q_vec += (q_inf(V_t) - q_vec) / tau_q * dt

            del dV_t_deterministic, dV_t_stochastic, V_t

            if t % 100 == 0:
                gc.collect()

            # Increment t
            t += 1


        ### Return output in a tensor
        return (V_mat, spks_mat)



#%% DEFINE SYNAPTIC NOISE CLASS

class synaptic_noise(object):

    def __init__(self, geo, sigma_e, tau_e, gio, sigma_i, tau_i):

        self.geo        = geo           # Mean excitatory conductance (uS)
        self.sigma_e    = sigma_e       # Variance parameter for excitatory conductance (uS)
        self.tau_e      = tau_e         # Correlation time of excitatory conductance (ms)

        self.gio        = gio
        self.sigma_i    = sigma_i
        self.tau_i      = tau_i

        self.ge         = None          # Matrix to hold time-varying excitatory conductuance
        self.gi         = None          # Matrix to hold time-varying inhibitory conductance


    def realize(self, shape, dt, seed = None):

        """
        Make stochastic time-varying excitatory and inhibitory conductances.
        """

        np.random.seed(None)
        rands_e = np.random.normal(size = shape)
        rands_i = np.random.normal(size = shape)

        realized_gs = self._realize(
        self.geo, self.sigma_e, self.tau_e,
        self.gio, self.sigma_i, self.tau_i,
        rands_e, rands_i, dt
        )

        self.ge = realized_gs[0, :, :]
        self.gi = realized_gs[1, :, :]

        return (self.ge, self.gi)


    @staticmethod
    @nb.jit(
        nb.float64[:, :, :](
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64[:, :],
        nb.float64[:, :],
        nb.float64
        )
    )
    def _realize(geo, sigma_e, tau_e, gio, sigma_i, tau_i, rands_e, rands_i, dt):

        # Create arrays to hold output
        ge = np.empty(rands_e.shape, dtype = np.float64)
        gi = np.empty(rands_i.shape, dtype = np.float64)

        # Set initial condition
        ge[:, 0] = geo
        gi[:, 0] = gio

        # Iterate over time
        for t in range(rands_e.shape[1] - 1):

            # Integrate ge
            dge_deterministic = (geo - ge[:, t]) * dt / tau_e
            dge_stochastic = sigma_e * np.sqrt(2 * tau_e) * rands_e[:, t] * np.sqrt(dt) / tau_e

            ge[:, t + 1] = ge[:, t] + dge_deterministic + dge_stochastic


            # Integrate gi
            dgi_deterministic = (gio - gi[:, t]) * dt / tau_i
            dgi_stochastic = sigma_i * np.sqrt(2 * tau_i) * rands_i[:, t] * np.sqrt(dt) / tau_i

            gi[:, t + 1] = gi[:, t] + dgi_deterministic + dgi_stochastic

        return np.array([ge, gi])



#%% DEFINE SIMULATION CLASSES

class simulation(Simulation):

    def __init__(self, I, V0, mod, replicates = 1, I_N = None, ge = None, Ee = None, gi = None, Ei = None, dt = 0.1):

        self._mod = deepcopy(mod) # Attach a copy of model just in case

        I, V_mat, spks_mat, dt = (
            self._mod.simulate(I, V0, replicates, I_N, ge, Ee, gi, Ei, dt = dt)
        )

        self.I      = I         # Injected current (nA)
        self.V      = V_mat     # Somatic voltage (mV)
        self.spks   = spks_mat  # Boolean vector of spks
        self.dt     = dt        # Simulation timestep


    ### Method to get replicates
    @property
    def replicates(self):

        inferred_replicates = [self.I.shape[0],
                               self.V.shape[0],
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

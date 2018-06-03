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


    def simulate(self, I, V0, dt = 0.1):

        """
        Simulate voltage.

        Returns:
            Tuple of I, V, I_g1, and spks in vector format.
        """

        mat = self._simulate(I, V0, self.C, self.g, self.g1, self.tau1, self.theta, self.reset, dt)

        V_vec = mat[0, :]
        I_g1 = mat[1, :]
        spks = mat[2, :]

        return (I, V_vec, I_g1, spks, dt)


    @staticmethod
    @nb.jit(nb.float64[:,:](
        nb.float64[:],
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64,
        nb.float64
        ))
    def _simulate(I, V0, C, g, g1, tau1, theta, reset, dt):

        """
        Private numba-accelerated method for simulation.
        Called by GIF_mod.simulate().
        """

        # Create vectors to store output
        V_vec = np.empty(len(I), dtype = np.float64)
        w_vec = np.empty(len(I), dtype = np.float64)
        spks = np.zeros(len(I), dtype = np.bool)

        # Set initial condition
        V_vec[0] = V0
        w_vec[0] = V_vec[0]

        # Integrate over time
        t = 0
        while t < (len(I) - 1):

            # Integrate conductance
            dw_t = (V_vec[t] - w_vec[t]) / tau1 * dt
            w_vec[t + 1] = w_vec[t] + dw_t

            # Spike if above threshold
            if V_vec[t] >= theta:
                spks[t] = True
                V_vec[t + 1] = reset
            else:
                # Integrate voltage
                dV_t = (-g * V_vec[t] - g1 * w_vec[t] + I[t]) / C * dt
                V_vec[t + 1] = V_vec[t] + dV_t

            # Increment t
            t += 1

        # Return output in a matrix
        return np.array([V_vec, g1 * w_vec, spks])


class simulation(object):

    def __init__(self, I, mod, V0 = 0, dt = 0.1):

        self._mod = deepcopy(mod) # Attach a copy of model just in case

        I, V_vec, I_g1, spks, dt = self._mod.simulate(I, V0, dt)

        self.I      = I         # Injected current (nA)
        self.V      = V_vec     # Somatic voltage (mV)
        self.I_g1   = I_g1      # Current passed by conductance g1 (nA)
        self.spks   = spks      # Boolean vector of spks
        self.dt     = dt        # Simulation timestep


    ### Methods to get various transformations of spiketrain
    def get_spk_vec(self):
        return self.spks

    def get_spk_inds(self):
        return np.where(self.spks)[0]

    def get_spk_times(self):
        return self.get_spk_inds() * self.dt


    ### Method to get time vector
    def get_t_vec(self):
        return np.arange(0, int(len(self.I) * self.dt), self.dt)


    ### Plotting methods
    def basic_plot(self):

        plt.figure(figsize = (8, 6))

        spec = plt.GridSpec(4, 1, height_ratios = [0.2, 1, 1, 0.5], hspace = 0.4)

        t = self.get_t_vec()

        I_plot = plt.subplot(spec[3, :])
        plt.plot(t, self.I)
        plt.ylabel('I (nA)')

        plt.subplot(spec[0, :], sharex = I_plot)
        plt.plot(t, self.spks)
        plt.ylabel('Spks')
        plt.ylim(-0.05, 1.05)

        plt.subplot(spec[1, :], sharex = I_plot)
        plt.plot(t, self.V)
        plt.ylabel('V (mV)')

        plt.subplot(spec[2, :], sharex = I_plot)
        plt.plot(t, self.I_g1)
        plt.ylabel('I_g1 (nA)')

        plt.show()

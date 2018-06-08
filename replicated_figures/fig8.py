#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import sys
sys.path.append('./src')
sys.path.append('./replicated_figures')

import RichCond as Cond
import pltools


#%% PERFORM SIMULATIONS

mod1 = Cond.model1()
mod2 = Cond.model2()

### Gain extraction
no_neurons = int(2e2)
V0 = -65
dt = 0.01
bin_width = 2

# Realize synaptic noise ahead of time.
noise_A = Cond.synaptic_noise(0.01, 0.0005, 3, 0.0085, 0.0005, 10)
noise_C.realize((2, 100000), dt)
noise_B = Cond.synaptic_noise(0.01, 0.01, 3, 0.015, 0.01, 10)
noise_C = Cond.synaptic_noise(0.01, 0.015, 3, 0.063, 0.015, 10)
noise_D = Cond.synaptic_noise(0.02, 0.01, 3, 0.012, 0.01, 10)

test_sim = Cond.simulation(0.5 * np.ones(100000), -65, mod1, replicates = 2, ge = noise_C.ge, Ee = 0, gi = noise_C.gi, Ei = -75, dt = dt)
test_sim.spks.sum()/2
test_sim.basic_plot()

#%%

class Analysis(object):

    """
    Container for freqs, gains, and phases extracted from a model.
    """

    def __init__(self):

        self.freqs = []
        self.gains = []
        self.phases = []

    def append(self, freq, gain, phase):

        self.freqs.append(freq)
        self.gains.append(gain)
        self.phases.append(phase)

    def convert_to_numpy(self):

        self.freqs = np.array(self.freqs)
        self.gains = np.array(self.gains)
        self.phases = np.array(self.phases)

    def convert_to_list(self):

        self.freqs = self.freqs.tolist()
        self.gains = self.gains.tolist()
        self.phases = self.phases.tolist()


    ### Method for performing firing rate gain analysis.
    def fr_gain(self, mod, synaptic_noise, no_neurons, V0, I0, I1, freqs, no_cycles = 20,
        discard_cycles = 5, bin_width = 2, plot = False, verbose = True, dt = 0.01):

        # Generate enough synaptic noise for the longest simulation.
        # Will be subsetted for shorter simulations to reduce compute time.
        no_noise_timesteps = no_cycles * 1e3 / (min(freqs) * dt)
        synaptic_noise.realize((no_neurons, no_noise_timesteps), dt)

        # Get gain at each frequency.
        for freq in freqs:

            if verbose:
                print('Simulating frequency {}Hz'.format(freq))

            t = np.arange(0, no_cycles * 1e3 / freq, dt)
            t = np.tile(t, (no_neurons, 1))

            oscillating_current = I1 * np.sin(t * 2 * np.pi * freq * 1e-3) + I0

            mod_sim = Cond.simulation(oscillating_current, V0, mod, no_neurons,
                ge = synaptic_noise.ge[:, :oscillating_current.shape[1]], Ee = 0,
                gi = synaptic_noise.gi[:, :oscillating_current.shape[1]], Ei = -75,
                dt = dt)

            gain, phase = mod_sim.extract_IO_gain_phase(freq, subthreshold = False,
                bin_width = bin_width, discard_cycles = discard_cycles, plot = plot)

            self.freqs.append(freq)
            self.gains.append(gain)
            self.phases.append(phase)

        self.convert_to_numpy()

        if verbose:
            print('Done!')


### Perform simulations.

analysis_A = Analysis()
analysis_B = Analysis()
analysis_C = Analysis()
analysis_D = Analysis()

analysis_A.fr_gain(mod1, noise_A, no_neurons, V0, I0, 0.02,
    freqs = np.linspace(2, 60, 20),
    no_cycles = 20, discard_cycles = 5, bin_width = bin_width, dt = dt)
analysis_B.fr_gain(mod1, noise_A, no_neurons, V0, I0, 0.2,
    freqs = np.linspace(2, 60, 20),
    no_cycles = 20, discard_cycles = 5, bin_width = bin_width, dt = dt)
analysis_C.fr_gain(mod1, noise_A, no_neurons, V0, I0, 0.1,
    freqs = np.linspace(2, 60, 20),
    no_cycles = 20, discard_cycles = 5, bin_width = bin_width, dt = dt)
analysis_D.fr_gain(mod2, noise_A, no_neurons, V0, I0, 0.1,
    freqs = np.linspace(2, 100, 20),
    no_cycles = 20, discard_cycles = 5, bin_width = bin_width, dt = dt)

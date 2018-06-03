#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./src')

import RichGIF as rGIF

#%% TEST MODEL CLASS

C       = 0.5 #nF
g       = 0.025 #uS
g1      = 0.025 #uS
tau1    = 100 #ms
theta   = 20 #mV
reset   = 14 #mV
dt      = 0.1 #ms

test_mod = rGIF.model(C, g, g1, tau1, theta, reset)

no_neurons = 10
t = np.arange(0, 1000, dt)
per = np.linspace(1000/2, 1000/7, len(t))
test_current = 0.024 * np.sin(t * 2 * np.pi / per) + 0.95
I_N = 0.11

I, V_mat, I_g1, spks, dt_ = test_mod.simulate(test_current, I_N, 0, no_neurons, dt)

# Make figure of simulation output
plt.figure(figsize = (10, 5))

t_mat = np.tile(t, (no_neurons, 1))

spks_plot = plt.subplot(411)
plt.plot(t_mat.T, spks.T, 'k-', alpha = 1/no_neurons)
plt.ylim(-0.05, 1.05)
plt.ylabel('Spike status')

plt.subplot(412, sharex = spks_plot)
plt.plot(t_mat.T, V_mat.T, 'k-', alpha = 1/no_neurons)
plt.ylabel('V (mV)')

plt.subplot(413, sharex = spks_plot)
plt.plot(t_mat.T, I_g1.T, 'k-', alpha = 1/no_neurons)
plt.ylabel('I_g1 (nA)')

plt.subplot(414, sharex = spks_plot)
plt.plot(t_mat.T, I.T, 'k-', alpha = 1/no_neurons)
plt.ylabel('I (nA)')
plt.xlabel('Time (ms)')

plt.tight_layout()
plt.show()

#%% TEST SIMULATION CLASS

test_sim = rGIF.simulation(test_current, I_N, test_mod, no_neurons)
test_sim.get_spk_mat()
test_sim.get_spk_inds()
test_sim.get_spk_times()
test_sim.basic_plot()

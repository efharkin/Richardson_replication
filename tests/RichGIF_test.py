#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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

no_neurons = 300
t = np.arange(0, 500, dt)
test_current = 0.024 * np.sin(t * 2 * np.pi / 200) + 0.95
V0 = 17.5

test_sim = rGIF.simulation(test_current, I_N, test_mod, no_neurons, V0)
test_sim.get_spk_mat()
test_sim.get_spk_inds()
test_sim.get_spk_times()
test_sim.basic_plot()

test_sim.get_firing_rate()

test_sim.firing_rate_plot('doc/img/fig1.png')

#%% TEST RETRIEVAL OF BINNED CURRENT AND FIRING RATE TOGETHER

bin_width = 5
bin_centres, binned_fr, binned_I = test_sim.get_firing_rate(bin_width = bin_width, return_I = True)

plt.figure(figsize = (10, 6))

plt.suptitle('Firing rate + current binning test')

ax = plt.subplot(211)
plt.bar(bin_centres, binned_fr, width = bin_width, edgecolor = 'k', facecolor = 'r')
plt.ylabel('Firing rate (Hz)')
ax.set_xticklabels([])

plt.subplot(212)
plt.bar(bin_centres, binned_I, width = bin_width, edgecolor = 'k', facecolor = 'none')
plt.xlabel('Time (ms)')
plt.ylabel('I (nA)')

plt.subplots_adjust(top = 0.9)
plt.show()

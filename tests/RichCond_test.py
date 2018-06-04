#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import sys
sys.path.append('./src')

import RichCond as Cond


#%% TEST MODEL 1 CLASS

test_mod = Cond.model1()

dt = 0.1
no_neurons = 3
t = np.arange(0, 1000, dt)
per = np.linspace(1000/2, 1000/7, len(t))
test_current = 0.024 * np.sin(t * 2 * np.pi / per) + 0.35
I_N = 0.11

I, V_mat, m_mat, h_mat, n_mat, Ihf_mat, Ihs_mat, spks, dt_ = test_mod.simulate(
    test_current, I_N, -60, no_neurons, -30, 2, dt
)


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
plt.plot(t_mat.T, m_mat.T, 'k-', alpha = 1/no_neurons, label = 'm')
plt.plot(t_mat.T, h_mat.T, 'r-', alpha = 1/no_neurons, label = 'h')
plt.plot(t_mat.T, n_mat.T, 'b-', alpha = 1/no_neurons, label = 'n')
plt.ylabel('g')
#plt.legend()

plt.subplot(414, sharex = spks_plot)
plt.plot(t_mat.T, I.T, 'k-', alpha = 1/no_neurons)
plt.ylabel('I (nA)')
plt.xlabel('Time (ms)')

plt.tight_layout()
plt.show()


#%% TEST MODEL 2 CLASS

test_mod2 = Cond.model2()

dt = 0.01
no_neurons = 3
t = np.arange(0, 1000, dt)
per = np.linspace(1000/2, 1000/7, len(t))
test_current = 0.024 * np.sin(t * 2 * np.pi / per) + 1.7
I_N = 0.11

I, V_mat, m_mat, h_mat, n_mat, p_mat, q_mat, spks, dt_ = test_mod2.simulate(
    test_current, I_N, -60, no_neurons, -30, 2, dt
)


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
plt.plot(t_mat.T, m_mat.T, 'k-', alpha = 1/no_neurons, label = 'm')
plt.plot(t_mat.T, h_mat.T, 'r-', alpha = 1/no_neurons, label = 'h')
plt.plot(t_mat.T, n_mat.T, 'b-', alpha = 1/no_neurons, label = 'n')
plt.ylabel('g')
#plt.legend()

plt.subplot(414, sharex = spks_plot)
plt.plot(t_mat.T, I.T, 'k-', alpha = 1/no_neurons)
plt.ylabel('I (nA)')
plt.xlabel('Time (ms)')

plt.tight_layout()
plt.show()


#%% TEST SIMULATION CLASS

dt = 0.01
no_neurons = 50
t = np.arange(0, 500, dt)
test_current = 0.024 * np.sin(t * 2 * np.pi / 200) + 0.3
V0 = -60

test_sim = Cond.simulation(test_current, I_N, test_mod, no_neurons, V0, dt)
test_sim.get_spk_mat()
test_sim.get_spk_inds()
test_sim.get_spk_times()
test_sim.basic_plot()

test_sim.get_firing_rate()

test_sim.firing_rate_plot()

test_sim2 = Cond.simulation(test_current, I_N, test_mod2, no_neurons, V0, dt)
test_sim2.firing_rate_plot()

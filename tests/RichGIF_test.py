#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./src')

import RichGIF as rGIF

#%% TEST SIMULATION

C       = 0.5 #nF
g       = 0.025 #uS
g1      = 0.025 #uS
tau1    = 100 #ms
theta   = 20 #mV
reset   = 14 #mV
dt      = 0.1 #ms

test_mod = rGIF.model(C, g, g1, tau1, theta, reset)

t = np.arange(0, 1000, dt)
per = np.linspace(1000/2, 1000/7, len(t))
test_current = 0.024 * np.sin(t * 2 * np.pi / per) + 0.95

I, V_vec, I_g1, spks, dt_ = test_mod.simulate(test_current, 0, dt)

# Make figure of simulation output
plt.figure(figsize = (10, 5))

spks_plot = plt.subplot(411)
plt.plot(t, spks)
plt.ylim(-0.05, 1.05)
plt.ylabel('Spike status')

plt.subplot(412, sharex = spks_plot)
plt.plot(t, V_vec)
plt.ylabel('V (mV)')

plt.subplot(413, sharex = spks_plot)
plt.plot(t, I_g1)
plt.ylabel('I_g1 (nA)')

plt.subplot(414, sharex = spks_plot)
plt.plot(t, I)
plt.ylabel('I (nA)')
plt.xlabel('Time (ms)')

plt.tight_layout()
plt.show()

#%% IMPORT MODULES

from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import sys
sys.path.append('./src')
sys.path.append('./replicated_figures')

import RichGIF as rGIF
import pltools


#%% PERFORM SIMULATIONS

### Illustrative protocol
C       = 0.5 #nF
g       = 0.025 #uS
g1      = 0.025 #uS
tau1    = 100 #ms
theta   = 20 #mV
reset   = 14 #mV
dt      = 0.1 #ms

mod = rGIF.model(C, g, g1, tau1, theta, reset)

no_neurons = int(5e3)
t = np.arange(0, 3000, dt)
oscillation_5Hz = np.sin(t[:10000] * 2 * np.pi / 200)
oscillation_20Hz = np.sin(t[:10000] * 2 * np.pi / 50)
illustrative_stimulus = np.concatenate(
    (np.zeros(10000), oscillation_5Hz, oscillation_20Hz),
    axis = -1
)
V0 = 17.5

low_noise_sim = rGIF.simulation(0.024 * illustrative_stimulus + 0.95, 0.11,
    mod, no_neurons, V0, dt)
high_noise_sim = rGIF.simulation(0.059 * illustrative_stimulus + 0.78, 0.55,
    mod, no_neurons, V0, dt)

low_noise_sim.firing_rate_plot(bin_width = 1)
high_noise_sim.firing_rate_plot(bin_width = 1)


### Gain extraction
no_neurons = int(2e2)
no_freqs = 15
V0 = 17.5
I_N = 0.55
bin_width = 2

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


high_noise_analysis = Analysis()
low_noise_analysis = Analysis()

for freq in np.logspace(0, 2):

    print('\rSimulating frequency {}'.format(freq), end = '')

    t = np.arange(0, 20 * 1e3 / freq, dt)
    oscillating_current = np.sin(t * 2 * np.pi * freq * 1e-3)

    low_noise_sim_2 = rGIF.simulation(0.024 * oscillating_current + 0.95, 0.11,
        mod, no_neurons, V0, dt)
    high_noise_sim_2 = rGIF.simulation(0.059 * oscillating_current + 0.78, 0.55,
        mod, no_neurons, V0, dt)

    gain_low, phase_low = low_noise_sim_2.extract_IO_gain_phase(freq, bin_width = bin_width, plot = False)
    low_noise_analysis.append(freq, gain_low, phase_low)

    gain_high, phase_high = high_noise_sim_2.extract_IO_gain_phase(freq, bin_width = bin_width, plot = False)
    high_noise_analysis.append(freq, gain_high, phase_high)


print('\nDone!')

low_noise_analysis.convert_to_numpy()
high_noise_analysis.convert_to_numpy()

low_noise_analysis.gains /= low_noise_analysis.gains[0]
high_noise_analysis.gains /= high_noise_analysis.gains[0]

#%% ASSEMBLE FIGURE

save_path = './doc/img/'

firing_rate_bin_width = 2
markersize = 1.2

plt.rc('text', usetex = True)

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

plt.figure(figsize = (6, 6.5))

spec = gs.GridSpec(3, 2, hspace = 0.6, wspace = 0.3, top = 0.95, bottom = 0.1, right = 0.95, height_ratios = [1.2, 1.2, 1])

specA = gs.GridSpecFromSubplotSpec(2, 1, spec[0, :], hspace = 0.6)
specA1 = gs.GridSpecFromSubplotSpec(2, 1, specA[0, :], height_ratios = [4, 1], hspace = 0)

specB = gs.GridSpecFromSubplotSpec(2, 1, spec[1, :], hspace = 0.6)
specB1 = gs.GridSpecFromSubplotSpec(2, 1, specB[0, :], height_ratios = [4, 1], hspace = 0)

### A: Low noise case
plt.subplot(specA1[0, :])
plt.title('\\textbf{{A1}} Low noise sample trace', loc = 'left')
t = low_noise_sim.get_t_vec()
V = deepcopy(low_noise_sim.V.T[:, 0])
V -= 70
V[low_noise_sim.get_spk_mat().T[:, 0]] = 0
plt.plot(t, V, 'k-', linewidth = 0.5)
pltools.add_scalebar(anchor = (0, 0.1), omit_x = True, y_units = 'mV')

plt.subplot(specA1[1, :])
plt.plot(t, 1000 * low_noise_sim.I.T[:, 0], 'k-', linewidth = 0.5)
pltools.add_scalebar(anchor = (0, 0.02), x_label_space = 0.1, bar_space = 0.04,
x_units = 'ms', y_units = 'pA', x_on_left = False, omit_x = True)

plt.subplot(specA[1, :])
plt.title('\\textbf{{A2}} Low noise mean firing rate', loc = 'left')
x, y = low_noise_sim.get_firing_rate(bin_width = firing_rate_bin_width)
plt.bar(x, y, width = firing_rate_bin_width, color = (0.8, 0.2, 0.2))
plt.text(1500, 45, '$f = $ resonance', ha = 'center', va = 'bottom')
plt.text(2500, 45, '$f = r_0$', ha = 'center', va = 'bottom')
plt.ylim(0, 65)
plt.xlabel('Time (ms)')
plt.ylabel('Rate (Hz)')


### B: High noise case
plt.subplot(specB1[0, :])
plt.title('\\textbf{{B1}} High noise sample trace', loc = 'left')
t = high_noise_sim.get_t_vec()
V = deepcopy(high_noise_sim.V.T[:, 0])
V -= 70
V[high_noise_sim.get_spk_mat().T[:, 0]] = 0
plt.plot(t, V, 'k-', linewidth = 0.5)
pltools.add_scalebar(anchor = (0, 0.1), omit_x = True, y_units = 'mV')

plt.subplot(specB1[1, :])
plt.plot(t, 1000 * high_noise_sim.I.T[:, 0], 'k-', linewidth = 0.5)
pltools.add_scalebar(anchor = (0, 0.02), x_label_space = 0.1, bar_space = 0.04,
x_units = 'ms', y_units = 'pA', x_on_left = False, omit_x = True)

plt.subplot(specB[1, :])
plt.title('\\textbf{{B2}} High noise mean firing rate', loc = 'left')
x, y = high_noise_sim.get_firing_rate(bin_width = firing_rate_bin_width)
plt.bar(x, y, width = firing_rate_bin_width, color = (0.2, 0.2, 0.8), aa = False)
plt.text(1500, 35, '$f = $ resonance', ha = 'center', va = 'bottom')
plt.text(2500, 35, '$f = r_0$', ha = 'center', va = 'bottom')
plt.ylim(0, 50)
plt.xlabel('Time (ms)')
plt.ylabel('Rate (Hz)')

ax = plt.subplot(spec[2, 0])
plt.title('\\textbf{{C1}} Signal gain', loc = 'left')
plt.axvline(5, linestyle = 'dotted', linewidth = 0.7, color = 'k')
plt.axvline(20, linestyle = '--', dashes = (10, 10), linewidth = 0.7, color = 'k')
ax.set_xscale('log')
plt.plot(low_noise_analysis.freqs, low_noise_analysis.gains, 'o',
    color = (0.8, 0.2, 0.2), markersize = markersize, label = 'Low noise')
plt.plot(high_noise_analysis.freqs, high_noise_analysis.gains, 'o',
    color = (0.2, 0.2, 0.8), markersize = markersize, label = 'High noise')
plt.annotate(
    'Resonance', xy = (5, 0.7), xytext = (-15, 0), textcoords = 'offset points',
    ha = 'right', va = 'center', arrowprops = {'arrowstyle': '->'}
)
plt.annotate(
    '$r_0$', xy = (20, 0.4), xytext = (15, -10), textcoords = 'offset points',
    ha = 'left', va = 'center', arrowprops = {'arrowstyle': '->'}
)
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')


ax = plt.subplot(spec[2, 1])
plt.title('\\textbf{{C2}} Signal phase-shift', loc = 'left')
ax.set_xscale('log')
plt.axvline(5, linestyle = 'dotted', linewidth = 0.7, color = 'k')
plt.axvline(20, linestyle = '--', dashes = (10, 10), linewidth = 0.7, color = 'k')
plt.plot(low_noise_analysis.freqs, 360 * low_noise_analysis.phases / (2 * np.pi), 'o',
    color = (0.8, 0.2, 0.2), markersize = markersize, label = 'Low noise')
plt.plot(high_noise_analysis.freqs, 360 * high_noise_analysis.phases / (2 * np.pi), 'o',
    color = (0.2, 0.2, 0.8), markersize = markersize, label = 'High noise')
plt.legend()
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase shift (degrees)')

if save_path is not None:
    plt.savefig(save_path + 'fig6.png', dpi = 300)

plt.show()

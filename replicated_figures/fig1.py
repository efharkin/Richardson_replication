#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import sys
sys.path.append('./src')
sys.path.append('./replicated_figures')

import RichGIF as rGIF
import pltools


#%% PERFORM SIMULATION

C       = 0.5 #nF
g       = 0.025 #uS
g1      = 0.025 #uS
tau1    = 100 #ms
theta   = 20 #mV
reset   = 14 #mV
dt      = 0.1 #ms

fig1_GIFmod = rGIF.model(C, g, g1, tau1, theta, reset)

no_neurons = 100
t = np.arange(0, 500, dt)
test_current = 0.024 * np.sin(t * 2 * np.pi / 200) + 0.95
V0 = 17.5
I_N = 0.11

fig1_GIFsim = rGIF.simulation(test_current, I_N, fig1_GIFmod, no_neurons, V0)


#%% ASSEMBLE FIGURE

bin_width = 10
save_path = './doc/img/fig1.png'

plt.rc('text', usetex = True)

plt.figure(figsize = (6, 6))

spec = gs.GridSpec(3, 1, height_ratios = [1.3, 1, 1], hspace = 0.4)
spec_inner = gs.GridSpecFromSubplotSpec(2, 1, spec[0, :], hspace = 0, height_ratios = [4, 1])


t_mat_transpose = fig1_GIFsim.get_t_mat().T

sample_neuron_plot = plt.subplot(spec_inner[0, :])
plt.title('\\textbf{{A.}} Sample trace', loc = 'left')
V_trace = fig1_GIFsim.V.T[:, 0] - 70
V_trace[fig1_GIFsim.spks[0, :]] = 0
plt.plot(t_mat_transpose[:, 0], V_trace, 'k-', linewidth = 0.5)
pltools.add_scalebar(anchor = (0, 0.1), omit_x = True, y_units = 'mV')

sample_current_plot = plt.subplot(spec_inner[1, :])
plt.plot(t_mat_transpose[:, 0], 1e3 * fig1_GIFsim.I.T[:, 0], 'k-', linewidth = 0.5)
pltools.add_scalebar(anchor = (0, 0.02), x_label_space = 0.1, bar_space = 0.04,
x_units = 'ms', y_units = 'pA', x_on_left = False)


raster_plot = plt.subplot(spec[1, :])
plt.title('\\textbf{{B.}} Raster plot', loc = 'left')
spk_times = fig1_GIFsim.get_spk_times()
for rep in range(fig1_GIFsim.replicates):
    plt.plot(spk_times[rep], [rep] * len(spk_times[rep]), '|', color = 'k', markersize = 1)
raster_plot.set_xticklabels([])
plt.ylabel('Neuron')

firing_rate_plot = plt.subplot(spec[2, :])
plt.title('\\textbf{{C.}} Mean firing rate', loc = 'left')
t_binned, binned_firing_rate = fig1_GIFsim.get_firing_rate(bin_width = bin_width)
plt.bar(t_binned, binned_firing_rate, width = bin_width, facecolor = 'none', edgecolor = 'k')
plt.axhline(binned_firing_rate.mean(),
linestyle = '--', dashes = (10, 10), color = 'k', linewidth = 0.7)
plt.annotate(
'$r_0$',
xy = (150, binned_firing_rate.mean()),
xytext = (0, 20), textcoords = 'offset points',
ha = 'center',
arrowprops = {'arrowstyle': '->'}
)
plt.ylabel('Rate (Hz)')
plt.xlabel('Time (ms)')

sample_neuron_plot.set_xlim(firing_rate_plot.get_xlim())
raster_plot.set_xlim(firing_rate_plot.get_xlim())

plt.tight_layout()

if save_path is not None:
    plt.savefig(save_path, dpi = 300)

plt.show()

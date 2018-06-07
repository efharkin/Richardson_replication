#%% IMPORT MODULES

import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import sys
sys.path.append('./src')
sys.path.append('./replicated_figures')

import RichCond as Cond
import pltools


#%% PERFORM SIMULATIONS

no_neurons = 1
V0 = -65
I_N = 0
bin_width = 0
dt = 0.1

# Settings related to length of simulation
no_cycles_high  = 50        # f > 10Hz
no_cycles_mid   = 20        # 10Hz >= f > 1Hz
no_cycles_low   = 10        # 1Hz >= f
discard_cycles = 5

do_plot1 = True
do_plot2 = False


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


mod1 = Cond.model1()
mod2 = Cond.model2()

mod1_analysis = Analysis()
mod2_analysis = Analysis()

for freq in np.logspace(-1, 3, 16):

    print('\rSimulating frequency {}'.format(freq), end = '')

    if freq > 10:
        no_cycles = no_cycles_high
    elif freq > 1:
        no_cycles = no_cycles_mid
    else:
        no_cycles = no_cycles_low

    t = np.arange(0, no_cycles * 1e3 / freq, dt)
    oscillating_current = 0.05 * np.sin(t * 2 * np.pi * freq * 1e-3)

    mod1_sim = Cond.simulation(oscillating_current + 0, V0, mod1, no_neurons, I_N = I_N, dt = dt)
    mod2_sim = Cond.simulation(oscillating_current + 0.4, V0, mod2, no_neurons, I_N = I_N, dt = dt)

    gain1, phase1 = mod1_sim.extract_IO_gain_phase(freq, subthreshold = True,
        bin_width = bin_width, discard_cycles = discard_cycles, plot = do_plot1)
    mod1_analysis.append(freq, gain1, phase1)

    gain2, phase2 = mod2_sim.extract_IO_gain_phase(freq, subthreshold = True,
        bin_width = bin_width, discard_cycles = discard_cycles, plot = do_plot2)
    mod2_analysis.append(freq, gain2, phase2)

mod1_analysis.convert_to_numpy()
mod2_analysis.convert_to_numpy()

for i in range(len(mod1_analysis.phases)):

    if mod1_analysis.phases[i] > np.pi:
        mod1_analysis.phases[i] -= 2 * np.pi
    if mod2_analysis.phases[i] > np.pi:
        mod2_analysis.phases[i] -= 2 * np.pi

print('\nDone!')

del mod1_sim, mod2_sim


#%% PICKLE ANALYSIS

PICKLE_PATH = './replicated_figures/'

with open(PICKLE_PATH + 'fig3_mod1analysis.pyc', 'wb') as f:
    pickle.dump(mod1_analysis, f)

with open(PICKLE_PATH + 'fig3_mod2analysis.pyc', 'wb') as f:
    pickle.dump(mod2_analysis, f)


#%% ASSEMBLE FIGURE

save_path = './doc/img/fig3.png'


plt.rc('text', usetex = True)

plt.figure(figsize = (6, 5))

spec = gs.GridSpec(2, 2, hspace = 0.5, wspace = 0.4)

mod1amp_plot = plt.subplot(spec[0, 0])
mod1amp_plot.set_xscale('log')
plt.title('\\textbf{{A1}} Model 1 impedance amplitude', loc = 'left')
plt.plot(mod1_analysis.freqs, mod1_analysis.gains, 'ko')
plt.annotate(
'$6.5$Hz resonance',
xy = (9, 24),
xytext = (30, 10), textcoords = 'offset points',
ha = 'center',
arrowprops = {'arrowstyle': '->'}
)
plt.ylim(0, 30)
plt.ylabel('Impedance (M$\Omega$)')
plt.xlabel('Frequency (Hz)')

mod1phase_plot = plt.subplot(spec[0, 1])
mod1phase_plot.set_xscale('log')
plt.title('\\textbf{{A2}} Model 1 phase shift', loc = 'left')
plt.plot(mod1_analysis.freqs, 360 * mod1_analysis.phases / (2 * np.pi), 'ko')
plt.ylabel('Phase shift (radians)')
plt.xlabel('Frequency (Hz)')

mod2amp_plot = plt.subplot(spec[1, 0])
mod2amp_plot.set_xscale('log')
plt.title('\\textbf{{B1}} Model 2 impedance amplitude', loc = 'left')
plt.plot(mod2_analysis.freqs, mod2_analysis.gains, 'ko')
plt.annotate(
'$30$Hz resonance',
xy = (35, 17),
xytext = (10, 20), textcoords = 'offset points',
ha = 'center',
arrowprops = {'arrowstyle': '->'}
)
plt.ylim(0, 25)
plt.ylabel('Impedance (M$\Omega$)')
plt.xlabel('Frequency (Hz)')

mod2phase_plot = plt.subplot(spec[1, 1])
mod2phase_plot.set_xscale('log')
plt.title('\\textbf{{B2}} Model 2 phase shift', loc = 'left')
plt.plot(mod2_analysis.freqs, 360 * mod2_analysis.phases / (2 * np.pi), 'ko')
plt.ylabel('Phase shift (radians)')
plt.xlabel('Frequency (Hz)')

if save_path is not None:
    plt.savefig(save_path, dpi = 300)

plt.show()

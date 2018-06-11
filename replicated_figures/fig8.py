#%% IMPORT MODULES

import pickle
import multiprocessing as mp
import gc

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import sys
import os
os.chdir('/Users/eharkin/Documents/School/Grad work/Courses/Computational summer school/Richardson_replication')
sys.path.append('./src')
sys.path.append('./replicated_figures')

import RichCond as Cond


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
noise_B = Cond.synaptic_noise(0.01, 0.01, 3, 0.015, 0.01, 10)


#%% DEFINE ANALYSIS CLASS

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

        # Get gain at each frequency.
        for freq in freqs:

            if no_cycles == 'default':

                if freq > 20:
                    no_cycles_internal = 50
                elif freq > 10:
                    no_cycles_internal = 20
                elif freq > 5:
                    no_cycles_internal = 10
                else:
                    no_cycles_internal = 5

            else:
                no_cycles_internal = no_cycles

            if discard_cycles == 'default':

                if freq > 20:
                    discard_cycles_internal = 15
                elif freq > 10:
                    discard_cycles_internal = 10
                elif freq > 5:
                    discard_cycles_internal = 5
                else:
                    discard_cycles_internal = 1

            else:
                discard_cycles_internal = discard_cycles

            if verbose:
                print('Simulating frequency {}Hz'.format(freq))

            t = np.arange(0, no_cycles_internal * 1e3 / freq, dt)
            oscillating_current = I1 * np.sin(t * 2 * np.pi * freq * 1e-3) + I0
            synaptic_noise.realize((no_neurons, len(oscillating_current)), dt)

            mod_sim = Cond.simulation(oscillating_current, V0, mod, no_neurons,
                ge = synaptic_noise.ge[:, :oscillating_current.shape[0]], Ee = 0,
                gi = synaptic_noise.gi[:, :oscillating_current.shape[0]], Ei = -75,
                dt = dt)

            if verbose:
                print('Extracting IO gain/phase.')
            gain, phase = mod_sim.extract_IO_gain_phase(freq, subthreshold = False,
                bin_width = bin_width, discard_cycles = discard_cycles_internal, plot = plot)

            self.freqs.append(freq)
            self.gains.append(gain)
            self.phases.append(phase)

            del mod_sim, t, oscillating_current
            gc.collect()

        self.convert_to_numpy()

        if verbose:
            print('Done!')

#%%

### Perform simulations.

no_points = 25

def worker(input_dict):

    """
    Worker function for multicore processing.
    """

    analysis_x = Analysis()
    analysis_x.fr_gain(
        input_dict['mod'],
        input_dict['noise'],
        input_dict['no_neurons'],
        input_dict['V0'],
        input_dict['I0'],
        input_dict['I1'],
        input_dict['freqs'],
        input_dict['no_cycles'],
        input_dict['discard_cycles'],
        input_dict['bin_width'],
        input_dict['plot'],
        input_dict['verbose'],
        input_dict['dt']
    )
    return analysis_x

no_cycles = 'default'
discard_cycles = 'default'

inputs = [
    {
    'mod': mod1,
    'noise': noise_A,
    'no_neurons': no_neurons,
    'V0': V0,
    'I0': 0.8,
    'I1': 0.02,
    'freqs': np.linspace(2, 60, no_points),
    'no_cycles': no_cycles,
    'discard_cycles': discard_cycles,
    'bin_width': bin_width,
    'plot': False,
    'verbose': True,
    'dt': dt
    },

    {
    'mod': mod1,
    'noise': noise_B,
    'no_neurons': no_neurons,
    'V0': V0,
    'I0': 0.7,
    'I1': 0.2,
    'freqs': np.linspace(2, 60, no_points),
    'no_cycles': no_cycles,
    'discard_cycles': discard_cycles,
    'bin_width': bin_width,
    'plot': False,
    'verbose': True,
    'dt': dt
    },

    {
    'mod': mod2,
    'noise': noise_A,
    'no_neurons': no_neurons,
    'V0': V0,
    'I0': 2,
    'I1': 0.1,
    'freqs': np.linspace(2, 100, no_points),
    'no_cycles': no_cycles,
    'discard_cycles': discard_cycles,
    'bin_width': bin_width,
    'plot': False,
    'verbose': True,
    'dt': dt
    },

    {
    'mod': mod2,
    'noise': noise_B,
    'no_neurons': no_neurons,
    'V0': V0,
    'I0': 1.2,
    'I1': 0.1,
    'freqs': np.linspace(2, 100, no_points),
    'no_cycles': no_cycles,
    'discard_cycles': discard_cycles,
    'bin_width': bin_width,
    'plot': False,
    'verbose': True,
    'dt': dt
    }
]

if __name__ == '__main__':
    p = mp.Pool(4)
    results = p.map(worker, inputs)

    with open('./replicated_figures/fig8_results_dump.pyc', 'wb') as f:
        pickle.dump(results, f)
        f.close()

#%% UNPICKLE SIMULATIONS

with open('./replicated_figures/fig8_results_dump.pyc', 'rb') as f:
    results = pickle.load(f)

mod1_noiseA = results[0]
mod1_noiseB = results[1]
mod2_noiseA = results[2]
mod2_noiseB = results[3]

del results

#%% COMPENSATE FOR >2PI PHASE SHIFTS

for mod in [mod1_noiseA, mod1_noiseB, mod2_noiseA, mod2_noiseB]:
    for i in range(len(mod.phases)):
        if mod.phases[i] > np.pi:
            mod.phases[i] -= 2 * np.pi


#%% MAKE FIGURE

save_path = './doc/img/fig8.png'

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

plt.figure(figsize = (6, 4))

spec = gs.GridSpec(2, 2, right = 0.95, top = 0.95, wspace = 0.35, hspace = 0.6)

# Mod 1
plt.subplot(spec[0, 0])
plt.title('\\textbf{{A1}} Model 1 gain', loc = 'left')
plt.axvline(6.5, linestyle = 'dotted', linewidth = 0.7, color = 'k')
plt.axvline(50, linestyle = '--', dashes = (10, 10), linewidth = 0.7, color = 'k')
plt.plot(mod1_noiseA.freqs, mod1_noiseA.gains / mod1_noiseA.gains.max(),
    'o', color = (0.8, 0.2, 0.2), markersize = markersize)
plt.plot(mod1_noiseB.freqs, mod1_noiseB.gains / mod1_noiseB.gains.max(),
    'o', color = (0.2, 0.2, 0.8), markersize = markersize)
plt.annotate(
    'Resonance', xy = (6.5, 0.35), xytext = (15, -10), ha = 'left', va = 'center',
    textcoords = 'offset points', arrowprops = {'arrowstyle': '->'}
)
plt.annotate(
    '$r_0$', xy = (50, 0.35), xytext = (-20, -5), ha = 'right', va = 'center',
    textcoords = 'offset points', arrowprops = {'arrowstyle': '->'}
)
plt.ylabel('Gain')
plt.xlabel('Frequency (Hz)')

plt.subplot(spec[0, 1])
plt.title('\\textbf{{A2}} Model 1 phase shift', loc = 'left')
plt.axvline(6.5, linestyle = 'dotted', linewidth = 0.7, color = 'k')
plt.axvline(50, linestyle = '--', dashes = (10, 10), linewidth = 0.7, color = 'k')
plt.plot(mod1_noiseA.freqs, 360 * mod1_noiseA.phases / (2 * np.pi),
    'o', color = (0.8, 0.2, 0.2), markersize = markersize, label = 'Low noise')
plt.plot(mod1_noiseB.freqs, 360 * mod1_noiseB.phases / (2 * np.pi),
    'o', color = (0.2, 0.2, 0.8), markersize = markersize, label = 'High noise')
plt.ylabel('Phase shift (radians)')
plt.xlabel('Frequency (Hz)')
plt.legend()

# Mod 2
plt.subplot(spec[1, 0])
plt.title('\\textbf{{B1}} Model 2 gain', loc = 'left')
plt.axvline(30, linestyle = 'dotted', linewidth = 0.7, color = 'k')
plt.axvline(50, linestyle = '--', dashes = (10, 10), linewidth = 0.7, color = 'k')
plt.plot(mod2_noiseA.freqs, mod2_noiseA.gains / mod2_noiseA.gains.max(),
    'o', color = (0.8, 0.2, 0.2), markersize = markersize)
plt.plot(mod2_noiseB.freqs, mod2_noiseB.gains / mod2_noiseB.gains.max(),
    'o', color = (0.2, 0.2, 0.8), markersize = markersize)
plt.ylabel('Gain')
plt.xlabel('Frequency (Hz)')

plt.subplot(spec[1, 1])
plt.title('\\textbf{{B2}} Model 2 phase shift', loc = 'left')
plt.axvline(30, linestyle = 'dotted', linewidth = 0.7, color = 'k')
plt.axvline(50, linestyle = '--', dashes = (10, 10), linewidth = 0.7, color = 'k')
plt.plot(mod2_noiseA.freqs, 360 * mod2_noiseA.phases / (2 * np.pi),
    'o', color = (0.8, 0.2, 0.2), markersize = markersize)
plt.plot(mod2_noiseB.freqs, 360 * mod2_noiseB.phases / (2 * np.pi),
    'o', color = (0.2, 0.2, 0.8), markersize = markersize)
plt.ylabel('Phase shift (radians)')
plt.xlabel('Frequency (Hz)')

if save_path is not None:
    plt.savefig(save_path, dpi = 300)

plt.show()

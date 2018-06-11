#%% IMPORT MODULES

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('./src')

import RichGIF as rGIF
import RichCond as Cond


### Set plotting parameters
save_path = './doc/img/'

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


#%% MAKE FIGURE FOR LIF MODEL

### LIF model
C       = 0.5 #nF
g       = 0.025 #uS
g1      = 0.025 #uS
tau1    = 100 #ms
theta   = 20 #mV
reset   = 14 #mV
dt      = 0.1 #ms

GIF_mod = rGIF.model(C, g, g1, tau1, theta, reset)

no_neurons = 5
t = np.arange(0, 500, dt)
per = np.linspace(1000/2, 1000/7, len(t))
test_current = 0.024 * np.sin(t * 2 * np.pi / per) + 0.95
I_N = 0.11

I, V_mat, I_g1, spks, dt_ = GIF_mod.simulate(test_current, I_N, 15, no_neurons, dt)


# Make figure of LIF model
plt.figure(figsize = (6, 4))

t_mat = np.tile(t, (no_neurons, 1))

plt.subplot(311)
plt.title('\\textbf{{A}} Subthreshold voltage', loc = 'left')
plt.plot(t_mat.T, V_mat.T, 'k-', linewidth = 0.5, alpha = 1/no_neurons)
plt.ylabel('$V$ (mV)')

plt.subplot(312)
plt.title('\\textbf{{B}} Current passed by $g_1 w_1$', loc = 'left')
plt.plot(t_mat.T, I_g1.T, 'k-', linewidth = 0.5, alpha = 1/no_neurons)
plt.ylabel('$I$ (nA)')

plt.subplot(313)
plt.title('\\textbf{{C}} Input stimulus', loc = 'left')
plt.plot(t_mat.T[:, 0], I.T[:, 0], 'k-')
plt.ylabel('$I_{{tot}}$ (nA)')
plt.xlabel('Time (ms)')

plt.tight_layout()

if save_path is not None:
    plt.savefig(save_path + 'appendix_gif.png', dpi = 300)

plt.show()


#%% MAKE FIGURE FOR COND MODEL WITH IH

mod1 = Cond.model1()

dt = 0.01
no_neurons = 2
t = np.arange(0, 500, dt)
per = np.linspace(1000/2, 1000/7, len(t))
test_current = 0.024 * np.sin(t * 2 * np.pi / per) + 0.35
I_N = 0.11

I, V_mat, spks, dt_ = mod1.simulate(
    test_current, -60, no_neurons, I_N = I_N, spk_detect_thresh = -30, spk_detect_tref = 2, dt = dt
)


### Make figure of mod1
plt.figure(figsize = (6, 4))

t_mat = np.tile(t, (no_neurons, 1))

plt.subplot(211)
plt.title('\\textbf{{A}} Voltage', loc = 'left')
plt.plot(t_mat.T, V_mat.T, 'k-', linewidth = 0.5, alpha = 1/no_neurons)
plt.ylabel('$V$ (mV)')

plt.subplot(212)
plt.title('\\textbf{{B}} Stimulus', loc = 'left')
plt.plot(t_mat.T[:, 0], I.T[:, 0], 'k-')
plt.ylabel('$I$ (nA)')
plt.xlabel('Time (ms)')

plt.tight_layout()

if save_path is not None:
    plt.savefig(save_path + 'appendix_mod1.png', dpi = 300)

plt.show()


#%% MAKE FIGURE FOR COND MODEL 2

mod2 = Cond.model2()

dt = 0.01
no_neurons = 2
t = np.arange(0, 500, dt)
per = np.linspace(1000/2, 1000/7, len(t))
test_current = 0.024 * np.sin(t * 2 * np.pi / per)
I_N = 0.11

I, V_mat, spks, dt_ = mod2.simulate(
    test_current + 1.7, -60, no_neurons, I_N = I_N, spk_detect_thresh = -30,
    spk_detect_tref = 2, dt = dt
)

I, V_mat_osc, spks_osc, dt_ = mod2.simulate(
    test_current + 1.5, -60, no_neurons, I_N = I_N, spk_detect_thresh = -30,
    spk_detect_tref = 2, dt = dt
)

### Make figure of mod2
plt.figure(figsize = (6, 4))

t_mat = np.tile(t, (no_neurons, 1))

plt.subplot(311)
plt.title('\\textbf{{A}} Firing with $I_0 = $ 1.7nA', loc = 'left')
plt.plot(t_mat.T, V_mat.T, 'k-', linewidth = 0.5, alpha = 1/no_neurons)
plt.ylabel('$V$ (mV)')

plt.subplot(312)
plt.title('\\textbf{{B}} Spontaneous oscillations with $I_0 = $ 1.5nA', loc = 'left')
plt.plot(t_mat.T, V_mat_osc.T, 'k-', linewidth = 0.5, alpha = 1/no_neurons)
plt.ylabel('$V$ (mV)')

plt.subplot(313)
plt.title('\\textbf{{C}} Oscillating component of stimulus ($I_{{sin}}$)', loc = 'left')
plt.plot(t_mat.T[:, 0], I.T[:, 0] - 1.5, 'k-')
plt.ylabel('$I$ (nA)')
plt.xlabel('Time (ms)')

plt.tight_layout()

if save_path is not None:
    plt.savefig(save_path + 'appendix_mod2.png', dpi = 300)

plt.show()

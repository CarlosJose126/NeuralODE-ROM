""" File to plot the results for the models """

import matplotlib.pyplot as plt
import scipy.linalg
import numpy as np
import pickle

from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import mean_squared_error


def POD(U, s_ind, e_ind, modes):
    """ Computes the spatial modes and temporal coefficients using the POD """

    # velocity in x
    S_ux = U[:, :, s_ind:e_ind, 0]
    S_ux = np.moveaxis(S_ux, [0, 1, 2], [1, 2, 0])

    # velocity in y
    S_uy = U[:, :, s_ind:e_ind, 1]
    S_uy = np.moveaxis(S_uy, [0, 1, 2], [1, 2, 0])

    # taking the temporal mean of snapshots
    S_uxm = np.mean(S_ux, axis=0)[np.newaxis, ...]
    S_uym = np.mean(S_uy, axis=0)[np.newaxis, ...]

    # fluctuating components: taking U-Um
    Ux = S_ux - S_uxm
    Uy = S_uy - S_uym
    
    Uxr = np.copy(Ux)
    Uyr = np.copy(Uy)

    # Reshaping to create snapshot matrix Y
    shape = Ux.shape
    Ux = Ux.reshape(shape[0], shape[1] * shape[2])
    Uy = Uy.reshape(shape[0], shape[1] * shape[2])
    Y = np.hstack((Ux, Uy))

    # Snapshot Method:
    Cs = np.matmul(Y, Y.T)

    # L:eigvals, As:eigvecs
    Lv, As = scipy.linalg.eigh(Cs)

    # descending order
    Lv = Lv[Lv.shape[0]::-1]
    As = As[:, Lv.shape[0]::-1]

    spatial_modes = np.matmul(Y.T, As[:, :modes]) / np.sqrt(Lv[:modes])
    temporal_coefficients = np.matmul(Y, spatial_modes)

    return spatial_modes, temporal_coefficients, Lv, Uxr, Uyr


def contour_plot(field, t_step):
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(20, 10)
    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.3, wspace=0.25)
    size_f = 20
    pad_f = 20

    for k in range(2):

        ax = fig.add_subplot(2, 1, k + 1)

        im = ax.imshow(field[k][t_step, :, :].T, origin='upper', cmap='jet')
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis='both', which='both', length=0)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.8)

        if k == 1:
            ax.set_title(r'$ROM$', rotation=0, size=size_f, pad=pad_f)

        else:
            ax.set_title(r'$FOM$', rotation=0, size=size_f, pad=pad_f)

        colorbar = plt.colorbar(im, cax=cax, ticks=np.linspace(-0.4, 0.4, 6))
        colorbar.ax.set_title(r'$m/s$', rotation=0, size=size_f, pad=pad_f)
        colorbar.ax.tick_params(labelsize=size_f)

        plt.setp(ax.spines.values(), linewidth=2)

    filename = 'Reconstruction_t{}'.format(t_step + 1)
    plt.savefig("%s.png" % (filename))
    # plt.close('all')


def modes_plot(predNODE, predLSTM, labelPOD, pod_modes, section):
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(40, 60)

    if section == 'first':
        t_plot = np.arange(100, 200)
    else:
        t_plot = np.arange(200, 300)

    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.3, wspace=0.25)

    v = int(pod_modes) + 1

    for k in range(pod_modes):

        ax = fig.add_subplot(v, 1, k + 1)
        ax.plot(t_plot, labelPOD[:, k], color='red', linewidth=9, alpha=1, label='POD')
        ax.plot(t_plot, predNODE[:, k], 'black', linestyle=':', linewidth=8, label='NODE')
        ax.plot(t_plot, predLSTM[:, k], 'deepskyblue', linestyle='--', linewidth=6, label='LSTM')

        ax.set_ylabel(r'$\alpha_{%d}$' % (k + 1), rotation=0, size=70, labelpad=10)

        if k == 0:
            ax.legend(bbox_to_anchor=(0.5, 1.4), loc='upper center', ncol=3, borderaxespad=0., fontsize=80)

        plt.setp(ax.spines.values(), linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=60)
        ax.tick_params(axis='both', which='minor', labelsize=60)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)
    ax.set_xlabel(r'$t$', size=60)
    filename = 'Prediction' + '_' + section
    plt.savefig("%s.png" % (filename))
    # plt.close('all')


def probe_plot(uX_probeLES, uX_probeNODE):
    fig, ax = plt.subplots(figsize=(40, 15))
    ax.plot(np.arange(0, 300), uX_probeLES, 'r', linewidth=6, label='LES')
    ax.plot(np.arange(0, 300), uX_probeNODE, 'k:', linewidth=6, label='NODE')
    ax.axvline(x=100, color='k', linewidth=5)

    plt.setp(ax.spines.values(), linewidth=4)
    ax.tick_params(axis='both', which='major', labelsize=70)
    ax.tick_params(axis='both', which='minor', labelsize=70)
    ax.xaxis.set_tick_params(width=4)
    ax.yaxis.set_tick_params(width=4)
    ax.set_ylim(-0.25, 0.25)
    ax.legend(bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=2, borderaxespad=0., fontsize=80)
    ax.set_ylabel(r'$u^{\prime}_x$', rotation=0, size=80, labelpad=30)
    ax.set_xlabel(r'$t$', size=60)
    filename = 'probe'
    plt.savefig("%s.png" % (filename))


# =============================================================================
'''                  Loading data and preprocessing                         '''
# =============================================================================

file_sim = open('./cylinderData.pkl', "rb")
data_LES = pickle.load(file_sim)
data_LES = np.nan_to_num(data_LES)

file_lstm = open('./data_lstm8.pth', "rb")
file_node = open('./data_node8.pth', "rb")

# temporal coefficients
predictions_LSTM = pickle.load(file_lstm).squeeze()
predictions_NODE = pickle.load(file_node).squeeze()

# define start and end times for POD
s_ind = 100
e_ind = data_LES.shape[2]
pod_modes = 8
spatial_modes, data_ROM, eigenvalues, Ux_LES, Uy_LES = POD(data_LES, s_ind, e_ind, pod_modes)

# percentage recovery with the first pod_modes
recovery_per = eigenvalues[:pod_modes] / eigenvalues.sum() * 100

# =============================================================================
'''  Evaluation of the forecast for the temporal coefficients in the test set  '''
# =============================================================================

# evaluate the error of the forecast:
mse_node = mean_squared_error(data_ROM[200:, ], predictions_NODE[200:, ])
mse_lstm = mean_squared_error(data_ROM[200:, ], predictions_LSTM[100:, :])

# plot the forecast of the temporal coefficients for the test data  100-200 t steps
modes_plot(predictions_NODE[100:200, ], predictions_LSTM[:100, :], data_ROM[100:200, ], 4, 'first')

# plot the forecast of the temporal coefficients for the test data  200-300 t steps
modes_plot(predictions_NODE[200:, ], predictions_LSTM[100:, :], data_ROM[200:, ], 4, 'second')

# reconstruction of the flow from the node data
Ur_n = np.matmul(predictions_NODE, spatial_modes.T)
# product Nx Ny
Nxy = data_LES.shape[0] * data_LES.shape[1]

# separate x and y velocities
Ur_x = Ur_n[:, :Nxy]
Ur_y = Ur_n[:, Nxy:]

# reshape to have Nt , Nx, Ny structure
shape = [data_LES.shape[2] - 100, data_LES.shape[0], data_LES.shape[1]]
Ux_NODE = Ur_x.reshape(shape[0], shape[1], shape[2])
Uy_NODE = Ur_y.reshape(shape[0], shape[1], shape[2])

# Plot reconstructions
Z = [Ux_LES, Ux_NODE]

for t in np.arange(100, 301, 50):
    contour_plot(Z, t - 1)

# Plot probe
uX_probeNODE = Ux_NODE[0:300, 127, 140]
uX_probeLES = Ux_LES[0:300, 127, 140]

probe_plot(uX_probeLES, uX_probeNODE)

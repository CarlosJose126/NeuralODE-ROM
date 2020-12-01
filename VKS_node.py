""" File with the random search for hyperparameters to reproduce the ROM-NN
    evolution of the temporal modes for the CYLINDER simulation:
"""
import scipy.linalg
import numpy as np
import time
import os
import gc

import matplotlib.pyplot as plt
from torchdiffeq import odeint
import torch.optim as optim
import torch.nn as nn
import argparse
import pickle
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--train_dir', type=str, default='./VKS_node_results')
parser.add_argument('--method', type=str, default='rk4')
parser.add_argument('--sched', type=eval, default=True)
args = parser.parse_args()


def set_seed(se):
    """ set the seeds to have reproducible results"""

    torch.manual_seed(se)
    torch.cuda.manual_seed_all(se)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(se)
    os.environ['PYTHONHASHSEED'] = str(se)


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

    return spatial_modes, temporal_coefficients


def normal_kl(mu1, lv1, mu2, lv2):
    """ Computes KL loss for VAE """

    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


class Encoder(nn.Module):
    """ Encoder :  transforms the input from data to latent
        space using a Seq2Vec architecture """

    def __init__(self, latent_dim, obs_dim, hidden_units, hidden_layers):
        super(Encoder, self).__init__()

        self.rnn = nn.GRU(obs_dim, hidden_units, hidden_layers, batch_first=True)
        self.h2o = nn.Linear(hidden_units, latent_dim * 2)
        

    def forward(self, x):

        y, _ = self.rnn(x)
        # take last step trough the dense layer
        y = y[:, -1, :]
        y = self.h2o(y)

        return y


class LatentOdeF(nn.Module):
    """ ODE-NN: takes the value z at the current time step and outputs the
        gradient dz/dt """

    def __init__(self, layers):
        super(LatentOdeF, self).__init__()

        self.act = nn.Tanh()
        self.layers = layers

        # Feedforward architecture
        arch = []
        for ind_layer in range(len(self.layers) - 2):
            layer = nn.Linear(self.layers[ind_layer], self.layers[ind_layer + 1])
            nn.init.xavier_uniform_(layer.weight)
            arch.append(layer)
        layer = nn.Linear(self.layers[-2], self.layers[-1])
        nn.init.xavier_uniform_(layer.weight)
        layer.weight.data.fill_(0)
        arch.append(layer)

        self.linear_layers = nn.ModuleList(arch)
        self.nfe = 0

    def forward(self, t, x):

        self.nfe += 1

        for ind in range(len(self.layers) - 2):
            x = self.act(self.linear_layers[ind](x))

        # last layer has identity activation (i.e linear)
        y = self.linear_layers[-1](x)
        return y


class Decoder(nn.Module):
    """ Decoder : transforms the input from latent to data space using a
        Seq2Seq architecture """

    def __init__(self, latent_dim, obs_dim, hidden_units, hidden_layers):
        super(Decoder, self).__init__()

        self.act = nn.Tanh()
        self.rnn = nn.GRU(latent_dim, hidden_units, hidden_layers, batch_first=True)
        self.h1 = nn.Linear(hidden_units, hidden_units - 5)
        self.h2 = nn.Linear(hidden_units - 5, obs_dim)

    def forward(self, x):
        y, _ = self.rnn(x)
        y = self.h1(y)
        y = self.act(y)
        y = self.h2(y)

        return y


def plotROM(predNODE, labelPOD, lossTrain, lossVal, itr, train_win, res_folder):
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(25, 15)
    fig.suptitle('Reconstruction of POD temporal modes using NODE - Epoch:  %04d' % itr, fontsize=24)

    filename = res_folder + '/' + str('%04d' % itr)

    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.45, wspace=0.25)

    # vector to define time axis in plots
    t_steps = np.linspace(0, 100, labelPOD.shape[0])

    for k in range(8):

        ax = fig.add_subplot(5, 2, k + 1)
        ax.plot(t_steps, labelPOD[:, k], color='r', linewidth=2.5, alpha=1, label='POD')
        ax.plot(t_steps, predNODE[0, :, k], 'k--', linewidth=2.5, label='NODE')
        ax.axvline(x=t_steps[train_win - 1], color='k')

        ax.set_ylabel('$a_{%d}$' % (k + 1), rotation=0, size=25, labelpad=10)

        if k == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=20)

        ax.set_xlabel(r'$t$', size=25)

        plt.setp(ax.spines.values(), linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)

    ax = fig.add_subplot(5, 2, 10)
    ax.plot(lossTrain, '-k', linewidth=2.0, label='Loss Train')
    ax.plot(lossVal, '--r', linewidth=2.0, label='Loss Validation')

    plt.xlabel('Epoch', fontsize=24)
    legend = ax.legend(loc=0, ncol=1, prop={'size': 20}, bbox_to_anchor=(0, 0, 1, 1), fancybox=True, shadow=False)
    plt.setp(legend.get_title(), fontsize='large')
    plt.setp(ax.spines.values(), linewidth=2)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    plt.savefig("%s.png" % (filename))
    plt.close('all')


# Make folder to save data (if not exists)
results_folder = args.train_dir
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

train_dir = './'

# Selecting gpu device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# =============================================================================
'''                  Loading data and preprocessing                         '''
# =============================================================================

sim_file = open('./cylinderData.pkl', "rb")
data_LES = pickle.load(sim_file)
data_LES = np.nan_to_num(data_LES)

# define start and end times for POD
s_ind = 100
e_ind = data_LES.shape[2]
pod_modes = 8
spatial_modes, data_ROM = POD(data_LES, s_ind, e_ind, pod_modes)

# window for training
twindow = int(0.25 * data_ROM.shape[0])

# temporal coefficients for training
data_ROM_t = data_ROM[:twindow, :]

# normalization
mean_data = data_ROM_t.mean(axis=0)
std_data = data_ROM_t.std(axis=0)
data_ROM_t = (data_ROM_t - mean_data) / std_data

# =============================================================================
'''                       Train and test data                              '''
# =============================================================================


# window for validation
vwindow = twindow + 25

# validation data
data_ROM_v = data_ROM[:vwindow, :]
data_ROM_v = (data_ROM_v - mean_data) / std_data
data_ROM_v = data_ROM_v.reshape((1, data_ROM_v.shape[0], data_ROM_v.shape[1]))
data_ROM_v = torch.FloatTensor(data_ROM_v).to(device)

# evaluation data
data_ROM_e = data_ROM[vwindow:, :]

# Reshaping: (batch, time steps, features)
data_ROM_t = data_ROM_t.reshape((1, data_ROM_t.shape[0], data_ROM_t.shape[1]))

# Convert to torch tensor
data_ROM_t = torch.FloatTensor(data_ROM_t).to(device)

# put data backward in time to infer z_0
idx = [i for i in range(data_ROM_t.size(0) - 1, -1, -1)]
idx = torch.LongTensor(idx).to(device)
obs_t = data_ROM_t.index_select(0, idx)

results_path = args.train_dir


def train(hyp_set, train_index):
    # create folder to save results
    results_folder = os.path.join(results_path, 'test_')
    results_folder += str(train_index)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # save training configuration
    results_data = {'hyperparameters': hyp_set}
    with open(results_folder + '/results_data.pth', 'wb') as f:
        pickle.dump(results_data, f)

    # =============================================================================
    '''                  Defining objects of the model                        '''
    # =============================================================================

    # feature dimension
    obs_dim = data_ROM_t.shape[2]
    # latent dimension
    latent_dim = obs_dim - hyp_set['latent_dim']

    # hidden units per layer in encoder
    units_enc = hyp_set['units_enc']
    # hidden layers encoder
    layers_enc = hyp_set['layers_enc']

    # layers in NODE block
    layers_node = [latent_dim] + list(hyp_set['layers_node']) + [latent_dim]
    # normalized vectors for ODE integration
    ts_ode = np.linspace(0, 1, data_ROM.shape[0])
    ts_ode = torch.from_numpy(ts_ode).float().to(device)
    ts_ode_t = ts_ode[:twindow]
    ts_ode_v = ts_ode[:vwindow]

    # hidden units per layer in decoder
    units_dec = hyp_set['units_dec']
    # hidden layers decoder
    layers_dec = hyp_set['layers_dec']

    # objects for VAE
    enc = Encoder(latent_dim, obs_dim, units_enc, layers_enc).to(device)
    node = LatentOdeF(layers_node).to(device)
    dec = Decoder(latent_dim, obs_dim, units_dec, layers_dec).to(device)

    # =============================================================================
    '''                       Training configurations                          '''
    # =============================================================================

    # Network's parameters
    params = (list(enc.parameters()) + list(node.parameters()) + list(dec.parameters()))

    optimizer = optim.AdamW(params, lr= hyp_set['lr'])

    # training loss metric using average
    loss_meter_t = RunningAverageMeter()
    # training loss metric without KL
    meter_train = RunningAverageMeter()
    # validation loss metric without KL
    meter_valid = RunningAverageMeter()

    # Scheduler for learning rate decay
    factor = 0.99
    min_lr = 1e-7
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=factor, patience=5, verbose=False, threshold=1e-5,
                                                     threshold_mode='rel', cooldown=0, min_lr=min_lr, eps=1e-08)

    criterion = torch.nn.MSELoss()

    # list to track  training losses
    lossTrain = []
    # list to track validation losses
    lossVal = []

    # number of iterations for the training
    iters = args.niters

    for itr in range(1, iters + 1):

        optimizer.zero_grad()

        # scheduler
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']
        if args.sched:
            scheduler.step(metrics=loss_meter_t.avg)

        out_enc = enc.forward(obs_t)
        # definition of mean and log var for codings
        qz0_mean, qz0_logvar = out_enc[:, :latent_dim], out_enc[:, latent_dim:]
        # noise
        epsilon = torch.randn(qz0_mean.size()).to(device)
        # sampling codings
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean
        # latent space evolution using node
        zt = odeint(node, z0, ts_ode_t, method=args.method).permute(1, 0, 2)
        output_vae_t = dec(zt)

        # compute KL loss
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        kl_loss = torch.mean(analytic_kl, dim=0)

        # VAE loss: MSE + KL
        loss = criterion(output_vae_t, data_ROM_t) + kl_loss

        # backpropagation
        loss.backward()
        # optimization step
        optimizer.step()
        # update training metric
        loss_meter_t.update(loss.item())
        # update training loss without KL
        meter_train.update(loss.item() - kl_loss.item())
        lossTrain.append(meter_train.avg)

        # validation step
        with torch.no_grad():

            enc.eval()
            node.eval()
            dec.eval()

            zv = odeint(node, z0, ts_ode_v, method=args.method).permute(1, 0, 2)
            output_vae_v = dec(zv)

            loss_v = criterion(output_vae_v[:, twindow:], data_ROM_v[:, twindow:])

            meter_valid.update(loss_v.item())
            lossVal.append(meter_valid.avg)

            enc.train()
            node.train()
            dec.train()

        if itr % 100 == 0:
            print('Iter: {}, Learning rate is: {:.4f}'.format(itr, current_lr))
            print('Iter: {}, Train Loss: {:.4f}'.format(itr, lossTrain[itr - 1]))
            print('Iter: {}, Valid Loss: {:.4f}'.format(itr, lossVal[itr - 1]))

            # scale output
            output_vae = (output_vae_v.cpu().detach().numpy()) * std_data + mean_data
            plotROM(output_vae, data_ROM[:vwindow, :], lossTrain, lossVal, itr, twindow, results_folder)

        if np.isnan(lossTrain[itr - 1]):
            break

    torch.save(enc.state_dict(), results_folder + '/enc.pth')
    torch.save(node.state_dict(), results_folder + '/node.pth')
    torch.save(dec.state_dict(), results_folder + '/dec.pth')

    # test results
    with torch.no_grad():

        enc.eval()
        node.eval()
        dec.eval()

        ze = odeint(node, z0, ts_ode, method=args.method).permute(1, 0, 2)
        output_vae_e = dec(ze)

        enc.train()
        node.train()
        dec.train()

    data_NODE = (output_vae_e.cpu().detach().numpy()) * std_data + mean_data

    with open('./data_node8.pth', 'wb') as f:
        pickle.dump(data_NODE, f)


# =============================================================================
'''                             Random Search                              '''
# =============================================================================

st = 1
n_samples = 1

for i in range(st, st + n_samples):
    gc.collect()

    set_seed(1234 + 8)

    start_e = int(time.time())

    hyp_set = {'latent_dim': np.random.randint(2, 5),
               'layers_enc': np.random.randint(1, 6),
               'units_enc': np.random.randint(10, 50),
               'layers_node': [np.random.randint(10, 50)] * np.random.randint(1, 3),
               'units_dec': np.random.randint(10, 50),
               'layers_dec': np.random.randint(1, 6),               
               'lr': round(10 ** np.random.uniform(-3.0, -1.0), 6)}

    print("Train #{}".format(i))
    print(hyp_set)
    train(hyp_set, i)
    stop_e = int(time.time())
    time_elapsed_e = stop_e - start_e

    print("Time elapsed in min {}".format(time_elapsed_e / 60))


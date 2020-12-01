""" File with the random search for hyper-parameters to reproduce the ROM-LSTM
    evolution of the temporal modes for the CYLINDER simulation:
"""

import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import scipy.linalg
import numpy as np

import argparse
import pickle
import torch
import os

parser = argparse.ArgumentParser()
parser.add_argument('--nepochs', type=int, default=200)
parser.add_argument('--train_dir', type=str, default='./VKS_lstm_results')
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


def plotROM(predLSTM, labelPOD, res_folder):
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(25, 15)
    fig.suptitle('Reconstruction of POD temporal modes using LSTM', fontsize=24)

    filename = res_folder + '/results_valid'
    t_steps = np.linspace(75, 100, 25)

    plt.subplots_adjust(top=0.9, bottom=0.05, hspace=0.35, wspace=0.25)
    
    # iterate over the 8 modes
    for k in range(8):

        ax = fig.add_subplot(5, 2, k + 1)
        ax.plot(t_steps, labelPOD[:, k], color='r', linewidth=2.5, label='POD')
        ax.plot(t_steps, predLSTM[:, k], 'k--', linewidth=2.5, label='LSTM')

        ax.set_ylabel('$a_{%d}$' % (k + 1), rotation=0, size=25, labelpad=10)

        if k == 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left',
                      borderaxespad=0., fontsize=20)

        ax.set_xlabel(r'$t$', size=25)
        plt.setp(ax.spines.values(), linewidth=2)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=12)
        ax.xaxis.set_tick_params(width=2)
        ax.yaxis.set_tick_params(width=2)

    plt.savefig("%s.png" % filename)
    plt.close('all')


class DataSet:

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        """return length of the dataset"""

        return len(self.features)

    def __getitem__(self, idx):
        """ The PyTorch DataLoader class will use this method to
            make an iterable for training or validation loops """

        feature = self.features[idx]
        label = self.labels[idx]

        return feature, label


class LSTM(nn.Module):
    """ Encoder : transforms the input from data to latent
        space using a Seq2Vec architecture """

    def __init__(self, output_dim, input_dim, hidden_units, hidden_layers):

        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_units, hidden_layers, batch_first=True)

        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.h2o = nn.Linear(hidden_units, output_dim)
        nn.init.xavier_uniform_(self.h2o.weight)

    def forward(self, x):

        y, _ = self.lstm(x)
        
        # take last step trough the dense layer
        y = y[:, -1, :]
        y = self.h2o(y)

        return y


class RunningAverageMeter(object):
    """ Computes and stores the average and current value of the losses """

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
        self.avg = 0
        self.val = None

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


# =============================================================================
'''                       Initial configurations                            '''
# =============================================================================


# Make folder to save data (if not exists)
results_folder = args.train_dir
if not os.path.exists(results_folder):
    os.makedirs(results_folder)

train_dir = './'

# check CUDA availability and set device
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# =============================================================================
'''                       Train and test data                              '''
# =============================================================================

batch_size = 15
seq_window = 10
total_size = data_ROM.shape[0] - seq_window

# divide the data using the seq_window
data_ROM_s = np.vstack([[data_ROM[t:t + seq_window, :] for t in range(total_size)]])
label_data_ROM_s = data_ROM[seq_window:, :]

# window for the training
twindow = int(0.25 * data_ROM.shape[0]) - seq_window

# training data
data_ROM_t = data_ROM_s[:twindow, :, :]
mean = data_ROM_t.reshape((-1, data_ROM_t.shape[2])).mean(axis=0)
std = data_ROM_t.reshape((-1, data_ROM_t.shape[2])).std(axis=0)
data_ROM_t = (data_ROM_t - mean) / std

label_ROM_t = label_data_ROM_s[:twindow, :]
label_ROM_t = (label_ROM_t - mean) / std

train_data = DataSet(data_ROM_t, label_ROM_t)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)

# validation and evaluation data
validation_labels = torch.FloatTensor((data_ROM[twindow + seq_window:][:25, :] - mean) / std)
validation_features = torch.FloatTensor((data_ROM[twindow:][:seq_window, :] - mean) / std).to(device)
evaluation_labels = torch.FloatTensor(data_ROM[twindow + seq_window:])

results_path = args.train_dir

# =============================================================================
'''                             Random Search                              '''
# =============================================================================

st = 1
n_samples = 1

for i in range(st, st + n_samples):

    train_index = i

    set_seed(1234 + 17)
    
    # hyperparameter configuration
    hyp_set = {'units': np.random.randint(10, 60),
               'layers': np.random.randint(1, 6),
               'lr': round(10 ** np.random.uniform(-3.0, -1.0), 6)}

    print("Train #{}".format(i))
    print(hyp_set)

    # output and input dimensions of the model
    out_dim = pod_modes
    inp_dim = pod_modes

    # units per layer and hidden layers
    units = hyp_set['units']
    layers = hyp_set['layers']

    # create folder to save results for this configuration
    results_folder = os.path.join(results_path, 'test_')
    results_folder += str(train_index)

    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # save configuration    
    results_data = dict(hyperparameters=hyp_set)

    with open(results_folder + '/results_data.pth', 'wb') as f:
        pickle.dump(results_data, f)

    # =============================================================================
    '''                  Defining objects of the model                        '''
    # =============================================================================

    model = LSTM(out_dim, inp_dim, units, layers).to(device)

    # =============================================================================
    '''                       Training configurations                          '''
    # =============================================================================

    lr = hyp_set['lr']
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=0.02)
    criterion = torch.nn.MSELoss()

    # number of iterations for the training
    epochs = args.nepochs

    # track change in validation loss
    valid_loss_min = np.Inf

    # Training loop   
    for epoch in range(1, epochs + 1):

        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        # train the model
        model.train()
        for data, target in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # compute predicted outputs by passing inputs to the model
            output = model(data.to(device))
            # calculate the batch loss
            loss = criterion(output, target.to(device))
            # compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item() * data.size(0)

        # validate the model  
        model.eval()
        with torch.no_grad():

            predictions = []

            data = validation_features.unsqueeze(axis=0)

            for _ in range(validation_labels.shape[0]):
                # compute predicted outputs by passing inputs to the model
                output = model(data)
                predictions.append(output.detach().cpu().numpy())
                # autoregressive step
                data = torch.cat((data, output.unsqueeze(axis=0)), 1)[-seq_window:, :, :]

            output = torch.FloatTensor(predictions).squeeze()
            # calculate the batch loss
            valid_loss = criterion(output, validation_labels)

        # calculate average losses
        train_loss = train_loss / len(train_loader.sampler)

        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min,
                valid_loss))
            torch.save(model.state_dict(), 'model_lstm.pt')
            valid_loss_min = valid_loss

    # load the best model
    model.load_state_dict(torch.load('model_lstm.pt'))

    # prep model for evaluation
    model.eval()

    # list to save predictions
    predictions = []
    # initial data for the autoregressive prediction
    data = validation_features.unsqueeze(axis=0)

    for _ in range(evaluation_labels.shape[0]):
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # save predictions
        predictions.append(output.detach().cpu().numpy())
        # autoregressive step      
        data = torch.cat((data, output.unsqueeze(axis=0)), 1)[-seq_window:, :, :]
        output = torch.FloatTensor(predictions).squeeze()

    test = evaluation_labels.cpu().detach().numpy()
    output_scaled = output.cpu().detach().numpy() * std + mean

    # save predictions
    with open('./data_lstm8.pth', 'wb') as f:
        pickle.dump(output_scaled[25:], f)

    # plot validation results
    plotROM(output_scaled[:25], test[:25], results_folder)

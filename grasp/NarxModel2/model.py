"""
DA-RNN model architecture.

@author Zhenye Na 05/21/2018
@modified 11/05/2019

References:
    [1] Yao Qin, Dongjin Song, Haifeng Chen, Wei Cheng, Guofei Jiang, Garrison W. Cottrell.
        "A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction"
        arXiv preprint arXiv:1704.02971 (2017).
    [2] Chandler Zuo. "A PyTorch Example to Use RNN for Financial Prediction" (2017).
"""

import matplotlib.pyplot as plt

import torch
import numpy as np

from torch import nn
from torch import optim

from torch.autograd import Variable
import torch.nn.functional as F


class Encoder(nn.Module):
    """encoder in DA_RNN."""

    def __init__(self, T,input_size,encoder_num_hidden,parallel=False):
        """Initialize an encoder in DA_RNN."""
        super(Encoder, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden #180
        self.input_size = input_size #180
        self.parallel = parallel
        self.T = T #20

        # Fig 1. Temporal Attention Mechanism: Encoder is LSTM
        self.encoder_lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.encoder_num_hidden,
            num_layers=1
        )

        # Construct Input Attention Mechanism via deterministic attention model
        # Eq. 8: W_e[h_{t-1}; s_{t-1}] + U_e * x^k
        self.encoder_attn = nn.Linear(
            in_features=2 * self.encoder_num_hidden + self.T,
            out_features=1
        )

    def forward(self, X):
        """forward.

        Args:
            X: input data

        """
        X_tilde = Variable(X.data.new(X.size(0), self.T, self.input_size).zero_())
        X_encoded = Variable(X.data.new(X.size(0), self.T, self.encoder_num_hidden).zero_())

        # Eq. 8, parameters not in nn.Linear but to be learnt
        # v_e = torch.nn.Parameter(data=torch.empty(
        #     self.input_size, self.T).uniform_(0, 1), requires_grad=True)
        # U_e = torch.nn.Parameter(data=torch.empty(
        #     self.T, self.T).uniform_(0, 1), requires_grad=True)
        # h_n, s_n: initial states with dimention hidden_size
        h_n = self._init_states(X) # ([1, 279, 180])
        s_n = self._init_states(X)

        for t in range(self.T - 1):
            # batch_size * input_size * (2 * hidden_size + T - 1)
            x = torch.cat((h_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           s_n.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           X.permute(0, 2, 1)), dim=2)

            x = self.encoder_attn(x.view(-1, self.encoder_num_hidden * 2 + self.T))


            # get weights by softmax
            alpha = F.softmax(x.view(-1, self.input_size), dim=1)

            # get new input for LSTM
            x_tilde = torch.mul(alpha, X[:, t, :])

            # Fix the warning about non-contiguous memory
            # https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            self.encoder_lstm.flatten_parameters()

            # encoder LSTM
            _, final_state = self.encoder_lstm(
                x_tilde.unsqueeze(0), (h_n, s_n))
            h_n = final_state[0]
            s_n = final_state[1]

            X_tilde[:, t, :] = x_tilde
            X_encoded[:, t, :] = h_n

        return X_tilde, X_encoded

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.encoder_num_hidden).zero_())


class Decoder(nn.Module):
    """decoder in DA_RNN."""

    def __init__(self, T, decoder_num_hidden, encoder_num_hidden):
        """Initialize a decoder in DA_RNN."""
        super(Decoder, self).__init__()
        self.decoder_num_hidden = decoder_num_hidden
        self.encoder_num_hidden = encoder_num_hidden
        self.T = T
        batch_size=279

        self.attn_layer = nn.Sequential(
            nn.Linear(2 * decoder_num_hidden +
                      encoder_num_hidden, encoder_num_hidden),
            nn.Tanh(),
            nn.Linear(encoder_num_hidden, 1)
        )
        self.tildet = nn.linear(self.T,batch_size)
        self.lstm_layer = nn.LSTM(
            input_size=1,
            hidden_size=decoder_num_hidden
        )
        self.fc = nn.Linear(self.decoder_num_hidden * 2, 1)
        self.fc_final = nn.Linear(decoder_num_hidden + encoder_num_hidden, 1)

        self.fc.weight.data.normal_()

    #def forward(self, X_encoded, y_prev):
    def forward(self, X_encoded, X_tilde):

        d_n = self._init_states(X_encoded)
        c_n = self._init_states(X_encoded)
        batch_size=X_encoded.shape[0]

        for t in range(self.T):
            x = torch.cat((d_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                           c_n.repeat(self.T, 1, 1).permute(1, 0, 2),
                           X_encoded), dim=2)

            beta = F.softmax(self.attn_layer(
                x.view(-1, 2 * self.decoder_num_hidden + self.encoder_num_hidden)).view(-1, self.T), dim=1)

            # Eqn. 14: compute context vector
            # batch_size * encoder_hidden_size
            context = torch.bmm(beta.unsqueeze(1), X_encoded)[:, 0, :]
            #if t < self.batchSize:
            # Eqn. 15
            # batch_size * 1
            #x_tilde=self.tildet(X_tilde[:,t, :].T).T # (279,180)
            y_tilde = self.fc(torch.cat((context, X_tilde[:,t, :]), dim=0).T) # attach 2d with 3D

            # Eqn. 16: LSTM
            # self.lstm_layer.flatten_parameters() # no effect on CPU
            decoder_output, final_states = self.lstm_layer(y_tilde.unsqueeze(0), (d_n, c_n))

            d_n = final_states[0]  # 1 * batch_size * decoder_num_hidden
            c_n = final_states[1]  # 1 * batch_size * decoder_num_hidden

        # get final predictin
        y_pred = self.fc_final(torch.cat((d_n[0], context), dim=1))

        return y_pred

    def _init_states(self, X):
        """Initialize all 0 hidden states and cell states for encoder."""
        # hidden state and cell state [num_layers*num_directions, batch_size, hidden_size]
        # https://pytorch.org/docs/master/nn.html?#lstm
        return Variable(X.data.new(1, X.size(0), self.decoder_num_hidden).zero_())


class DA_RNN(nn.Module):
    """Dual-Stage Attention-Based Recurrent Neural Network."""

    def __init__(self, trainx, trainy, testx, testy, T,encoder_num_hidden,decoder_num_hidden,learning_rate,epochs,parallel=False):
        """initialization."""
        super(DA_RNN, self).__init__()
        self.encoder_num_hidden = encoder_num_hidden # 180
        self.decoder_num_hidden = decoder_num_hidden # 180
        self.learning_rate = learning_rate
        self.parallel = parallel
        self.shuffle = False
        self.epochs = epochs
        self.T = T
        self.trainx = trainx # (180, 299, 133)
        self.trainy = trainy - np.mean(trainy,axis=0) # (299, 133)
        self.testx = testx
        self.testy = testy - np.mean(testy,axis=0)
        self.totleLen=trainx.shape[1] #299
        self.batch_size = self.totleLen-self.T #279
        self.input_size = self.trainx.shape[0] #180

        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print("==> Use accelerator: ", self.device)

        self.Encoder = Encoder(input_size=trainx.shape[0],
                               encoder_num_hidden=encoder_num_hidden,
                               T=T).to(self.device)
        self.Decoder = Decoder(encoder_num_hidden=encoder_num_hidden,
                               decoder_num_hidden=decoder_num_hidden,
                               T=T).to(self.device)

        # Loss function
        self.criterion = nn.MSELoss()

        if self.parallel:
            self.encoder = nn.DataParallel(self.encoder)
            self.decoder = nn.DataParallel(self.decoder)

        self.encoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Encoder.parameters()),
                                            lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(params=filter(lambda p: p.requires_grad,
                                                          self.Decoder.parameters()),
                                            lr=self.learning_rate)


    def train(self):
        """Training process."""
        #iter_per_epoch = int(np.ceil(self.train_timesteps * 1. / self.batch_size))
        #self.iter_losses = np.zeros(self.epochs * iter_per_epoch)
        #self.epoch_losses = np.zeros(self.epochs)
        for epoch in range(self.epochs):
            totalLen=self.trainx.shape[1]
            #ref_idx = np.array(range(totalLen - self.T))

            #idx = 0
            #trial=0
            # loop through all trials
            y_predicts=[]
            y_trues=[]
            for trial in range(self.trainx.shape[2]):
                trialdata=self.trainx[:,:,trial]
                targetdata=self.trainy[:,trial]
                y_trues.append(targetdata)

                x = np.zeros((self.batch_size, self.input_size, self.T))
                y_trueTemp = targetdata[self.T:] #(279,)

                # format 1 trial into 3D tensor
                for bs in range(self.batch_size):
                    x[bs, :, :] = trialdata[:,bs:bs+self.T]
                    #y_prev[bs, :] = self.y[indices[bs]: (indices[bs] + self.T - 1)]
                x=x.swapaxes(1,2)
                loss, y_predict = self.train_forward(x, y_trueTemp)
                #self.iter_losses[int(epoch * iter_per_epoch + idx / self.batch_size)] = loss

                trial += 1
                if epoch % 10000 == 0 and epoch != 0:
                    for param_group in self.encoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9
                    for param_group in self.decoder_optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * 0.9

                #self.epoch_losses[epoch] = np.mean(self.iter_losses[range(epoch * iter_per_epoch, (epoch + 1) * iter_per_epoch)])
            y_predicts.append(y_predict)
            #if epoch % 10 == 0:
            print("Epochs: ", epoch,
                  " Loss: ", self.epoch_losses[epoch])

            if epoch % 5 == 0:
                fig, ax = plt.subplots(figsize=(6, 3))
                plt.ion()
                ax.clear()
                #ax.plot(range(1, 1 + len(self.y)), self.y, label="True")
                ax.plot(y_trues, label="True", linewidth=0.1)
                #ax.plot(range(self.T, len(y_train_pred) + self.T),y_train_pred, label='Predicted - Train')
                ax.plot(y_predicts, label='Prediction',linewidth=0.1)
                ax.legend(loc='upper left')
                figname = '/Users/long/BCI/python_scripts/models/NARX/result/predOnTrain'+str(epoch)+'.pdf'
                fig.savefig(figname)
                plt.close(fig)

    def train_forward(self, X,y_trueTemp): # process one trial 3D data
        """Forward pass."""
        # zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_weighted, input_encoded = self.Encoder(
            Variable(torch.from_numpy(X).type(torch.FloatTensor).to(self.device)))
        y_pred = self.Decoder(input_encoded, input_weighted)

        y_true = Variable(torch.from_numpy(y_trueTemp).type(torch.FloatTensor).to(self.device))

        y_true = y_true.view(-1, 1)
        loss = self.criterion(y_pred, y_true)
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item(),y_pred

    def test(self, on_train=False):
        """Prediction."""
        y_preds = []
        y_trus = []

        for trial in range(self.testx.shape[2]):
            trialdata = self.testx[:, :, trial]
            targetdata = self.testy[:, trial]
            y_trus.append(targetdata)
            x = np.zeros((self.batch_size, self.input_size, self.T))
            y_trueTemp = targetdata[self.T:]  # (279,)

            # format 1 trial into 3D tensor
            for bs in range(self.batch_size):
                x[bs, :, :] = trialdata[:, bs:bs + self.T]
                # y_prev[bs, :] = self.y[indices[bs]: (indices[bs] + self.T - 1)]
            x = x.swapaxes(1, 2)
            input_weighted, input_encoded = self.Encoder(Variable(torch.from_numpy(x).type(torch.FloatTensor).to(self.device)))
            y_pred = self.Decoder(input_encoded,input_weighted).cpu().data.numpy()[:, 0]
            y_preds.append(y_pred)

            # plot the prediction
            fig, ax = plt.subplots(figsize=(6, 3))
            plt.ion()
            ax.clear()
            # ax.plot(range(1, 1 + len(self.y)), self.y, label="True")
            ax.plot(y_trus, label="True", linewidth=0.1)
            # ax.plot(range(self.T, len(y_train_pred) + self.T),y_train_pred, label='Predicted - Train')
            ax.plot(y_preds, label='Prediction', linewidth=0.1)
            ax.legend(loc='upper left')
            figname = '/Users/long/BCI/python_scripts/models/NARX/result/predOnTest' + str(epoch) + '.pdf'
            fig.savefig(figname)
            plt.close(fig)
        return y_preds

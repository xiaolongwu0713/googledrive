import torch
from skorch.callbacks import Callback
from skorch import NeuralNetRegressor
from skorch.utils import to_tensor
import matplotlib.pyplot as plt
import numpy as np

from grasp.config import *


class plotPrediction(Callback):
    def __init__(self,result_dir):
        self.result_dir=result_dir
    def on_epoch_begin(self, net,dataset_train=None, dataset_valid=None, **kwargs):
        preds.clear()
        targets.clear()

    def on_batch_end(self, net, X=None, y=None, training=None, **kwargs):
        if training==False:
            target=y.squeeze().numpy()
            step=kwargs
            loss=step['loss']
            y_pred=step['y_pred']
            y_pred=y_pred.squeeze().cpu().detach().numpy()
            preds.append(y_pred)
            targets.append(target)

    def on_epoch_end(self, net,dataset_train=None, dataset_valid=None, **kwargs):
        fig, ax = plt.subplots(figsize=(6, 3))
        plt.ion()
        ax.clear()
        a=np.concatenate(targets)
        b=np.concatenate(preds)
        ax.plot(a, label="True", linewidth=1)
        ax.plot(b, label='Predicted - Test', linewidth=1)
        ax.legend(loc='upper left')
        figname = self.result_dir + 'prediction' + str(len(net.history)) + '.png'
        fig.savefig(figname)
        plt.close(fig)

class MyRegressor(NeuralNetRegressor):
    def get_loss(self, y_pred, y_true, X=None, training=False):
        loss = super().get_loss(y_pred, y_true, X=X, training=training)
        #l1 = 0.01
        #Lambda = 1e-6
        # only specific weight
        #loss += l1 * self.module_.layer0.weight
        # all weights
        loss2 = Lambda * sum([w.abs().sum() for w in self.module_.parameters()])
        loss=loss+loss2
        return loss

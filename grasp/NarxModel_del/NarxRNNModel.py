import sys
#sys.path.insert(0,'..') or
sys.path.append('../..')

import argparse
from grasp.NarxModel.model import *
from grasp.NarxModel.utils import read_data

'''
usage: cd to grasp folder, then type 'python NarxRNNModel.py' in terminal
or cd to grasp folder, type python NarxRNNModel.py --help
'''

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="PyTorch implementation of paper 'A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction'")

    # Dataset setting
    parser.add_argument('--dataroot', type=str, default="../nasdaq/nasdaq100_padding.csv", help='path to dataset')
    parser.add_argument('--feature', type=str, default="fbands", help='input feature: fbands or raw')
    parser.add_argument('--traindataset', type=int, default=4, help='which movement dataset to train on')
    parser.add_argument('--testdataset', type=int, default=4, help='which movement dataset to test for')
    parser.add_argument('--batchsize', type=int, default=128, help='input batch size [128]')

    # Encoder / Decoder parameters setting
    parser.add_argument('--nhidden_encoder', type=int, default=128, help='size of hidden states for the encoder m [64, 128]')
    parser.add_argument('--nhidden_decoder', type=int, default=128, help='size of hidden states for the decoder p [64, 128]')
    parser.add_argument('--ntimestep', type=int, default=10, help='the number of time steps in the window T [10]')

    # Training parameters setting
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    """Main pipeline of DA-RNN."""
    args = parse_args()

    # Read dataset
    print("==> Load dataset ...")
    X, y = read_data(args, debug=False)

    # Initialize model
    print("==> Initialize DA-RNN model ...")
    model = DA_RNN(X,y,args.ntimestep,args.nhidden_encoder,args.nhidden_decoder,args.batchsize,args.lr,args.epochs)

    # Train
    print("==> Start training ...")
    model.trainAndTest()

    # Prediction
    #model.eval()
    y_pred = model.test()

    #fig1 = plt.figure()
    #plt.semilogy(range(len(model.iter_losses)), model.iter_losses)
    #plt.savefig("/Users/long/BCI/python_scripts/models.bak/NARX/NARX_iter_loss.png")
    #plt.close(fig1)

    #fig2 = plt.figure()
    #plt.semilogy(range(len(model.epoch_losses)), model.epoch_losses)
    #plt.savefig("/Users/long/BCI/python_scripts/models.bak/NARX/NARX_epoch_loss.png")
    #plt.close(fig2)
    #print('Finished Training')


if __name__ == '__main__':
    main()

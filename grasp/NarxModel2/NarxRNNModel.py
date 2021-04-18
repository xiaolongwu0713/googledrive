import sys
#sys.path.insert(0,'..') or
sys.path.append('../..')

import argparse
from grasp.NarxModel2.model import *
#from grasp.NarxModel2.utils import read_data
from grasp.utils import read_fbanddata

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
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate [0.001] reduced by 0.1 after each 10000 iterations')

    # parse the arguments
    args = parser.parse_args()

    return args

def main():
    T = 20  # use last T time to do the prediction
    batchSize=128
    epochs=1
    learnRate=0.001
    encoderHiddenStateSize=180
    decoderHiddenStateSize=180

    """Main pipeline of DA-RNN."""
    args = parse_args() # not use any more

    # Read dataset
    print("==> Load dataset ...")
    #X, y = read_data(args, debug=False)
    train, test = read_fbanddata()
    trainx=train[:-2,:,:]
    trainy=train[180,:,:]
    testx = test[:-2, :, :]
    testy = test[180, :, :]
    #trainx,trainy,testx, testy = read_fbanddata(split=True)
    #train = np.reshape(train.swapaxes(1, 2), (182, -1))
    #test = np.reshape(test.swapaxes(1, 2), (182, -1))
    #X=np.concatenate((train,test),axis=1)
    #X=X[:-2,:] # (180, 42159)
    #y=X[-2,:] #(42159,)

    # Initialize model
    print("==> Initialize DA-RNN model ...")
    model = DA_RNN(trainx,trainy,T,encoderHiddenStateSize,decoderHiddenStateSize,learnRate,epochs)

    # Train
    print("==> Start training ...")
    model.train()

if __name__ == '__main__':
    main()

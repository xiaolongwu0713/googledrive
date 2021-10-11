import random
import numpy as np
import torch
import os
from prettytable import PrettyTable

# data is an epoch data( after .get_data() call.)
def slide_epochs(epoch, label, wind, stride):
    total_len = epoch.shape[2]
    X = []
    Xi = []
    for trial in epoch:  # (63, 2001)
        s = 0
        while stride * s + wind < total_len:
            start = s * stride
            tmp = trial[:, start:(start + wind)]
            Xi.append(tmp)
            s = s + 1
        # add the last window
        last_s = s - 1
        if stride * last_s + wind < total_len - 100:
            tmp = trial[:, -wind:]
            Xi.append(tmp)

    X.append(Xi)
    X = np.concatenate(X, axis=0)  # (1300, 63, 500)
    samples_number = len(X)
    y = [label] * samples_number

    return X, y





import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from lgrt4gps.lgrtn import LGRTN
from scipy.io import loadmat
import urllib.request
import os
from tqdm.contrib import tzip

def test_sarcos():
    infile = "sarcos_inv_train.mat"
    if not os.path.exists(infile):
        print("Downloading Sarcos training data...")
        urllib.request.urlretrieve(
            "http://www.gaussianprocess.org/gpml/data/sarcos_inv.mat", infile)
    data_tr = loadmat(infile)['sarcos_inv'].astype(np.float32)
    xtr, ytr = data_tr[:,:21], data_tr[:,21:]

    infile = "sarcos_inv_test.mat"
    if not os.path.exists(infile):
        print("Downloading Sarcos test data...")
        urllib.request.urlretrieve(
            "http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat", infile)
    data_te = loadmat(infile)['sarcos_inv_test'].astype(np.float32)
    xte, yte = data_te[:,:21], data_te[:,21:]
    dx, dy = xtr.shape[1], ytr.shape[1]

    print("Data successfully loaded")

    lgrt_gp = LGRTN(dx, dy)

    for (x,y) in tzip(xtr,ytr):
        lgrt_gp.add_data(x,y)
        mu, s2 = lgrt_gp.predict(xte)


if __name__ == '__main__':
    test_sarcos()
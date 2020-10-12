import numpy as np
from time import time
import pandas as pd
from lgrt4gps.lgrtn import LGRTN
from scipy.io import loadmat
from scipy.stats import norm

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
    xtr, ytr = data_tr[:,:21], data_tr[:,21:22]

    infile = "sarcos_inv_test.mat"
    if not os.path.exists(infile):
        print("Downloading Sarcos test data...")
        urllib.request.urlretrieve(
            "http://www.gaussianprocess.org/gpml/data/sarcos_inv_test.mat", infile)
    data_te = loadmat(infile)['sarcos_inv_test'].astype(np.float32)
    xte, yte = data_te[:,:21], data_te[:,21:22]
    dx, dy = xtr.shape[1], ytr.shape[1]

    print("Data successfully loaded")

    lgrt_gp = LGRTN(dx, dy)
    labels = ['N','t_train', 't_pred', 'tree_depth', 'num_leaves', 'mse','nll']
    results = [[[]]*len(labels)]
    i = 0
    for (x,y) in tzip(xtr,ytr):
        x, y = x.reshape(1,dx), y.reshape(1,dy)
        t0 = time()
        lgrt_gp.add_data(x,y)
        t_train = time() - t0
        t0 = time()
        mu, s2 = lgrt_gp.predict(xte)
        t_pred = time() - t0
        if i % 100 == 0:
            results.append([lgrt_gp.ndata, t_train, t_pred/xte.shape[0],
                            lgrt_gp.depth, lgrt_gp.num_leaves ,
                            np.mean((mu - yte) ** 2),
                            -np.sum(norm.logpdf(yte,loc=mu, scale=np.sqrt(s2))) ])
        i += 1

    df = pd.DataFrame(results, columns=labels)
    df.to_csv('results_sarcos.csv')
if __name__ == '__main__':
    test_sarcos()
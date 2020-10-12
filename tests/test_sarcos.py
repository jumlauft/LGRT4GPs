import numpy as np
from timeit import default_timer as timer
import pandas as pd
import cProfile
import pstats
from lgrt4gps.lgrtn import LGRTN
from scipy.io import loadmat
from scipy.stats import norm

import urllib.request
import os
from tqdm import trange
profile = cProfile.Profile()
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

    # lgrt_gp = LGRTN(dx, dy)
    labels = ['N','t_train', 't_pred','t_add',
              'tree_height', 'num_leaves',
              'mse','nll']
    results = []
    #lgrt_gp.add_data(xtr[-40000:,:], ytr[-40000:,:])
    ntr = xtr.shape[0]
    for i in trange(10000,ntr,10000):
        t0 = timer()
        lgrt_gp = LGRTN(dx, dy)
        t1 = timer()
        idx = np.random.choice(ntr, i, replace=False)
        idx1 = np.random.choice(ntr, 1, replace=False)
        t2 = timer()
        lgrt_gp.add_data(xtr[idx,:],ytr[idx,:])
        t3 = timer()
        mu, s2 = lgrt_gp.predict(xte)
        t4 = timer()
        lgrt_gp.add_data(xtr[idx1, :], ytr[idx1, :])
        t5 = timer()

        results.append([i, t3-t2, (t4-t3)/xte.shape[0], t5-t4,
                            lgrt_gp.height, lgrt_gp.leaf_count ,
                            np.mean((mu - yte) ** 2),
                            -np.sum(norm.logpdf(yte,loc=mu, scale=np.sqrt(s2))) ])

    df = pd.DataFrame(results, columns=labels)
    df.to_csv('results_sarcos.csv')

if __name__ == '__main__':
    test_sarcos()

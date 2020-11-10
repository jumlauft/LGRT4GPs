import numpy as np
import sys, os

sys.path.append(os.path.abspath('./'))
from lgrt4gps.lgrtn import LGRTN
from sklearn.gaussian_process import GaussianProcessRegressor as GP
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


def test_init():
    LGRTN(1, 1)

    # Unavailable GP engine
    try:
        LGRTN(1, 1, GP_engine='any')
    except NotImplementedError:
        pass

def test_add_data():
    # single node
    nte, ntr = 1, 1000
    dx, dy = 3, 2
    np.random.seed(0)
    xtr = np.random.uniform(0, 1, size=(ntr, dx))
    ytr = np.random.uniform(0, 1, size=(ntr, dy))

    # Check idea of lazy training
    root = LGRTN(dx, dy, div_method='median')
    root.add_data(xtr, ytr)
    root.leaf_count
    LGRTN(dx, dy, div_method='mean').add_data(xtr, ytr)
    LGRTN(dx, dy, div_method='center').add_data(xtr, ytr)





def test_fit():
    # single node
    nte, ntr = 1, 10
    dx, dy = 1, 1
    xtr = np.random.uniform(0, 1, size=(ntr, dx))
    ytr = np.random.uniform(0, 1, size=(ntr, dy))

    # Check idea of lazy training
    lgrtn = LGRTN(dx, dy, lazy_training=True, optimize_hyps=False)
    lgrtn.add_data(xtr, ytr)
    assert lgrtn._update_gp == True
    lgrtn.fit()
    assert lgrtn._update_gp == False

    # Verify failure of hyp optimizer for simple
    lgrtn = LGRTN(dx, dy, lazy_training=True, optimize_hyps=True)
    lgrtn.add_data(xtr, ytr)
    try:
        lgrtn.fit()
    except NotImplementedError:
        pass

    # Verfiy hyps are not updated if not wanted
    lgrtn = LGRTN(dx, dy, lazy_training=False, optimize_hyps=False)
    lgrtn.add_data(xtr, ytr)
    hyps = (lgrtn._gps[0].kernel_.get_params()['k1__k1__constant_value'],
            lgrtn._gps[0].kernel_.get_params()['k1__k2__length_scale'],
            lgrtn._gps[0].kernel_.get_params()['k2__noise_level'])
    lgrtn.fit()
    new_hyps = (lgrtn._gps[0].kernel_.get_params()['k1__k1__constant_value'],
            lgrtn._gps[0].kernel_.get_params()['k1__k2__length_scale'],
            lgrtn._gps[0].kernel_.get_params()['k2__noise_level'])
    assert all([o == n for o,n in zip(hyps, new_hyps)])

    # Verfiy hyps are updated if  wanted
    lgrtn = LGRTN(dx, dy, lazy_training=False, optimize_hyps=False)
    lgrtn.add_data(xtr, ytr)
    hyps = (lgrtn._gps[0].kernel_.get_params()['k1__k1__constant_value'],
            lgrtn._gps[0].kernel_.get_params()['k1__k2__length_scale'],
            lgrtn._gps[0].kernel_.get_params()['k2__noise_level'])
    lgrtn.fit(optimize_hyps=True)
    new_hyps = (lgrtn._gps[0].kernel_.get_params()['k1__k1__constant_value'],
                lgrtn._gps[0].kernel_.get_params()['k1__k2__length_scale'],
                lgrtn._gps[0].kernel_.get_params()['k2__noise_level'])
    assert all([o != n for o, n in zip(hyps, new_hyps)])


def test_predict():
    np.random.seed(0)
    nte, ntr = 100, 1000
    dx, dy = 2, 1
    xtr = np.random.uniform(0, 1, size=(ntr, dx))
    ytr = np.sum(xtr ** 2, axis=1).reshape(-1,1)
    xte = np.random.uniform(0, 1, size=(nte, dx))
    yte = np.sum(xte ** 2, axis=1).reshape(-1,1)

    gpr = GP(kernel=ConstantKernel(1.0) * RBF((1.0,)*dx) + WhiteKernel(0.01),
             optimizer=None)
    gpr.fit(xtr, ytr)
    muteGP, sigteGP = gpr.predict(xte, return_std=True)

    lgrtn = LGRTN(dx, dy, inf_method='moe', max_pts=ntr)
    lgrtn.add_data(xtr, ytr)

    mute = lgrtn.predict(xte)
    assert np.mean((muteGP - mute) ** 2) < 1e-10

    mute, sigte = lgrtn.predict(xte, return_std=True)

    assert np.mean((muteGP - mute) ** 2) < 1e-10
    assert np.mean((sigteGP.reshape(-1,1) - sigte) ** 2) < 1e-10

    lgrtn = LGRTN(dx, dy, inf_method='moe', max_pts=100)
    lgrtn.add_data(xtr, ytr)
    mute, sigte = lgrtn.predict(xte, return_std=True)
    assert np.mean((muteGP - mute) ** 2) < 1e-4
    assert np.mean((sigteGP.reshape(-1,1)  - sigte) ** 2) < 1e-4


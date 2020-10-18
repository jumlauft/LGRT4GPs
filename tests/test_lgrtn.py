import numpy as np
import sys, os

sys.path.append(os.path.abspath('./'))
from lgrt4gps.lgrtn import LGRTN


def test_init():
    LGRTN(1, 1, GP_engine='')
    LGRTN(1, 1, GP_engine='GPy')

    # Unavailable GP engine
    try:
        LGRTN(1, 1, GP_engine='any')
    except NotImplementedError:
        pass

    # Missmatch of kernel input dimension
    from lgrt4gps.kern import RBF
    try:
        LGRTN(2, 1, kerns=[RBF(input_dim=3)])
    except ValueError:
        pass

    # Missmatch of kernel count
    from lgrt4gps.kern import RBF
    try:
        LGRTN(1, 2, kerns=[RBF(input_dim=1)])
    except AssertionError:
        pass

    # Missmatch of kernel class with GP engine
    from GPy.kern import RBF
    try:
        LGRTN(1, 1, kerns=[RBF(input_dim=1)])
    except TypeError:
        pass


def test_add_data():
    # single node
    nte, ntr = 1, 1000
    dx, dy = 3, 2
    xtr = np.random.uniform(0, 1, size=(ntr, dx))
    ytr = np.random.uniform(0, 1, size=(ntr, dy))

    # Check idea of lazy training
    LGRTN(dx, dy, GP_engine='', div_method='median').add_data(xtr, ytr)
    LGRTN(dx, dy, GP_engine='', div_method='mean').add_data(xtr, ytr)
    LGRTN(dx, dy, GP_engine='', div_method='center').add_data(xtr, ytr)

    LGRTN(dx, dy, GP_engine='GPy', div_method='median').add_data(xtr, ytr)
    LGRTN(dx, dy, GP_engine='GPy', div_method='mean').add_data(xtr, ytr)
    LGRTN(dx, dy, GP_engine='GPy', div_method='center').add_data(xtr, ytr)


def test_fit():
    # single node
    nte, ntr = 1, 10
    dx, dy = 1, 1
    xtr = np.random.uniform(0, 1, size=(ntr, dx))
    ytr = np.random.uniform(0, 1, size=(ntr, dy))

    # Check idea of lazy training
    lgrtn = LGRTN(dx, dy, GP_engine='', lazy_training=True, optimize_hyps=False)
    lgrtn.add_data(xtr, ytr)
    assert lgrtn._update_gp == True
    lgrtn.fit()
    assert lgrtn._update_gp == False

    # Verify failure of hyp optimizer for simple
    lgrtn = LGRTN(dx, dy, GP_engine='', lazy_training=True, optimize_hyps=True)
    lgrtn.add_data(xtr, ytr)
    try:
        lgrtn.fit()
    except NotImplementedError:
        pass

    # Verfiy hyps are not updated if not wanted
    lgrtn = LGRTN(dx, dy, GP_engine='GPy', lazy_training=True,
                  optimize_hyps=False)
    lgrtn.add_data(xtr, ytr)
    hyps = [lgrtn.kernels[0].parameters[0].values[0],
            lgrtn.kernels[0].parameters[1].values[0]]
    lgrtn.fit()
    new_hyps = [lgrtn.kernels[0].parameters[0].values[0],
                lgrtn.kernels[0].parameters[1].values[0]]
    assert hyps[0] == new_hyps[0]
    assert hyps[1] == new_hyps[1]

    # Verfiy hyps are updated if  wanted
    lgrtn = LGRTN(dx, dy, GP_engine='GPy', lazy_training=True,
                  optimize_hyps=True)
    lgrtn.add_data(xtr, ytr)
    hyps = [lgrtn.kernels[0].parameters[0].values[0],
            lgrtn.kernels[0].parameters[1].values[0]]
    lgrtn.fit()
    new_hyps = [lgrtn.kernels[0].parameters[0].values[0],
                lgrtn.kernels[0].parameters[1].values[0]]
    assert hyps[0] != new_hyps[0]
    assert hyps[1] != new_hyps[1]


def test_predict():
    np.random.seed(0)
    nte, ntr = 100, 10000
    dx, dy = 2, 2
    xtr = np.random.uniform(0, 1, size=(ntr, dx))
    ytr = xtr ** 2
    xte = np.random.uniform(0, 1, size=(nte, dx))
    yte = xte ** 2

    lgrtn = LGRTN(dx, dy, GP_engine='', inf_method='moe')
    lgrtn.add_data(xtr, ytr)
    mute1, _ = lgrtn.predict(xte)

    lgrtn = LGRTN(dx, dy, GP_engine='GPy', inf_method='moe')
    lgrtn.add_data(xtr, ytr)
    mute2, _ = lgrtn.predict(xte)

    assert np.mean((yte - mute1) ** 2) < 0.01
    assert np.mean((yte - mute2) ** 2) < 0.01

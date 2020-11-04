import numpy as np
import sys, os

sys.path.append(os.path.abspath('./'))
from lgrt4gps.gp import GP
from lgrt4gps.kern import RBF

def test_init():
    n, dx = 10, 2
    x = np.random.rand(n,dx)
    y = np.random.rand(n)
    gp = GP(x, y, RBF(dx))

    n, dx = 0, 2
    x = np.empty((0,dx))
    y = np.empty((0,))
    gp = GP(x, y, RBF(dx))

def test_predict():
    nte = 20
    n, dx = 10, 2
    x = np.random.rand(n,dx)
    y = np.random.rand(n)
    yte, s2 = GP(x, y, RBF(dx)).predict(np.random.rand(nte,dx))
    assert yte.shape == (nte,1)
    assert s2.shape == (nte,1)

    n, dx = 0, 2
    x = np.empty((0,dx))
    y = np.empty((0,))
    yt, s2 = GP(x, y, RBF(dx)).predict(np.random.rand(nte,dx))
    ytz = np.zeros((nte,1))
    assert np.all(yt==ytz)
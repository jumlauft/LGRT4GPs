import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from src.lgrtn import LGRTN
import GPy
from time import time


def test_1D():
    # set input / output dimnesions
    dx, dy = 1, 1
    # number of training /test  points
    ntr, nte = int(500/2), 100
    f = lambda x: np.sin(x[:, :1])

    # generate training data
    xtr = np.vstack((np.random.uniform(1, 3,size=(ntr, dx)),
                         np.random.uniform(7, 9, size=(ntr, dx))))
    ytr = f(xtr)
    # generate test data
    xte = np.linspace(0, 10, nte).reshape(-1,1)
    yte = f(xte)

    # process LGRT_GP
    LGRT_GP = LGRTN(dx, dy)
    LGRT_GP.add_data(xtr,ytr)
    mu_LGRT_GP, s2_LGRT_GP = LGRT_GP.predict(xte)

    # display tree structure
    print(LGRT_GP)

    # print results
    mse_LGRT_GP = np.mean((mu_LGRT_GP - yte) ** 2)
    nll_LGRT_GP = -np.sum(norm.logpdf(yte,loc=mu_LGRT_GP, scale=np.sqrt(s2_LGRT_GP)))
    print('LDGP: MSE: {:f}'.format(mse_LGRT_GP) + ', NLL: {:f}'.format(nll_LGRT_GP))

    # process full GP
    GP = GPy.models.GPRegression(xtr, ytr, GPy.kern.RBF(input_dim=dx))
    mu_fullGP, s2_fullGP = GP.predict(xte)

    # print results
    MSE_fullGP = np.mean((mu_fullGP - yte) ** 2)
    NLL_fullGP = -np.sum(norm.logpdf(yte,loc=mu_fullGP, scale=np.sqrt(s2_fullGP)))
    print('fullGP: MSE: {:f}'.format(MSE_fullGP) + ', NLL: {:f}'.format(NLL_fullGP))

    beta = 0.2
    plt.figure()
    plt.scatter(xtr, ytr)
    plt.plot(xte, mu_LGRT_GP,'r')
    plt.plot(xte, mu_fullGP,'b')
    plt.plot(xte, mu_LGRT_GP + beta*s2_LGRT_GP,'r--')
    plt.plot(xte, mu_fullGP + beta*s2_fullGP, 'b--')
    plt.legend(['mu LGRT_GP', 'mu fullGP',  's2 LGRT_GP', 's2 fullGP', 'Training data',])
    plt.plot(xte, mu_fullGP - beta*s2_fullGP,'b--')
    plt.plot(xte, mu_LGRT_GP - beta*s2_LGRT_GP, 'r--')
    plt.show()


def test_2D():
    # set input / output dimensions
    dx, dy = 2, 2
    # number of training /test  points
    ntr, nte = 10000, 1000
    f = lambda x: np.hstack((np.sin(x[:, :1]) + np.cos(x[:, 1:]),
                           (x[:, :1] + x[:, 1:])**2))

    # generate training data
    xtr = np.random.uniform(0, 10, size=(ntr, dx))
    ytr = f(xtr)
    # generate test data
    xte =  np.random.uniform(0, 10, size=(nte, dx))
    yte = f(xte)

    # process LGRT_GP
    t0 = time()
    LGRT_GP = LGRTN(dx, dy, max_pts=500)
    LGRT_GP.add_data(xtr, ytr)
    tt_LDGP = time() - t0
    t0 = time()
    mu_LGRT_GP, s2_LGRT_GP = LGRT_GP.predict(xte)
    ti_LDGP = time() - t0

    # print results
    mse_LGRT_GP = np.mean((mu_LGRT_GP - yte) ** 2)
    nll_LGRT_GP = -np.sum(norm.logpdf(yte,loc=mu_LGRT_GP, scale=np.sqrt(s2_LGRT_GP)))
    print('LGRT_GP: MSE: {:f}'.format(mse_LGRT_GP) + ', NLL: {:f}'.format(nll_LGRT_GP))
    print('        Time training {:f}'.format(tt_LDGP) + ', Time inference {:f}'.format(ti_LDGP))

    # process full GP
    t0 = time()
    GP = GPy.models.GPRegression(xtr, ytr, GPy.kern.RBF(input_dim=dx))
    tt_fullGP = time() - t0
    t0 = time()
    mu_fullGP, s2_fullGP = GP.predict(xte)
    ti_fullGP = time() - t0

    # print results
    MSE_fullGP = np.mean((mu_fullGP - yte) ** 2)
    NLL_fullGP = -np.sum(norm.logpdf(yte,loc=mu_fullGP, scale=np.sqrt(s2_fullGP)))
    print('fullGP: MSE: {:f}'.format(MSE_fullGP) + ', NLL: {:f}'.format(NLL_fullGP))
    print('        Time training {:f}'.format(tt_fullGP) + ', Time inference {:f}'.format(ti_fullGP))



def main():
    test_1D()
    test_2D()
if __name__ == '__main__':
    main()
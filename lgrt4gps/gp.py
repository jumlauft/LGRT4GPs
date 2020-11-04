import numpy as np
from scipy.linalg import solve_triangular, cholesky

def priorZero(x):
    return np.zeros((x.shape[0],1))

class GP:
    def __init__(self, x, y, kernel, sigon=0.01, prior=priorZero):
        # Check input
        n, dx = x.shape
        y = y.reshape(-1,1)
        if y.shape != (n,1) or dx != kernel.input_dim:
            raise ValueError('dimension mismatch')


        self.X = x
        self.kernel = kernel
        self.prior = prior
        self.dx = dx
        if n > 0:
            K = kernel(x, x)
            self.L = cholesky(K + (sigon ** 2) * np.eye(n))
            self.alpha = solve_triangular(self.L,
                                          solve_triangular(self.L, y-prior(x),
                                                           trans=1))


    def predict(self, Xt):
        if Xt.ndim == 2 != 2 or Xt.shape[1] != self.dx:
            raise ValueError('dimension mismatch')

        if self.X.shape[0] > 0:
            Kxxt = self.kernel(self.X, Xt)
            mu = np.dot(self.alpha.reshape(-1), Kxxt).reshape(-1, 1) + self.prior(Xt)
            v = solve_triangular(self.L, Kxxt, trans=1)
            s2 = self.kernel(Xt) - np.sum(v ** 2, axis=0)  # np.diag(np.dot(v.T,v))#
        else:
            mu = self.prior(Xt)
            s2 = self.kernel(Xt)
        return mu, s2.reshape(-1, 1)

    def optimize(self, **kwargs):
        raise NotImplementedError

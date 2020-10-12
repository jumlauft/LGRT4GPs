import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve_triangular, cholesky

class Kernel:
    def __init__(self, name='SE', input_dim=1):
        self.name = name
        self.dx = input_dim
        if name == 'SEard':
            self.hyps = np.ones(self.dx+1)
        elif name == 'SE':
            self.hyps = np.ones(1+1)
        else:
            raise NotImplementedError

    def __call__(self, *nargs):
        if len(nargs) == 2:
            xa, xb = nargs
            if self.name == 'SE':
                return self.hyps[0] ** 2 *\
                       (np.exp(-0.5 * (cdist(xa, xb)) / self.hyps[1] ** 2))
            else:
                raise NotImplementedError
        elif len(nargs) == 1:
            x = nargs[0]
            if self.name == 'SE':
                return self.hyps[0] ** 2 * np.ones(x.shape[0])
            else:
                raise NotImplementedError
        else:
            raise TypeError
class GP:
    def __init__(self, X, Y, kernel):
        sigon = 1
        self.X = X
        n,dx = X.shape
        self.kernel = kernel
        K = kernel(X,X)
        self.L = cholesky(K + (sigon**2) * np.eye(n))
        self.alpha = solve_triangular(self.L, solve_triangular(self.L,Y, trans = 1))

    def predict(self, Xt):
        Kxxt = self.kernel(self.X, Xt)
        mu = np.dot(self.alpha.reshape(-1), Kxxt)
        v = solve_triangular(self.L, Kxxt, trans=1)
        s2 = self.kernel(Xt) - np.sum(v**2, axis=0) # np.diag(np.dot(v.T,v))#
        return mu.reshape(-1,1), s2.reshape(-1,1)



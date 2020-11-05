import numpy as np
from scipy.spatial.distance import cdist
import warnings
class Kernel:
    def __init__(self, input_dim):
        self.input_dim = input_dim
    def __call__(self, *nargs):
        warnings.warn('Kernel is non-callable super class, use RBF')


class RBF(Kernel):
    def __init__(self, input_dim, hyps=None):
        super().__init__(input_dim)
        if hyps is None:
            hyps = np.ones((2,))
        self.hyps = hyps

    def __call__(self, *nargs):
        if len(nargs) == 2:
            xa, xb = nargs
            return self.hyps[0] ** 2 *\
                       (np.exp(-0.5 * (cdist(xa, xb)**2) / self.hyps[1] ** 2))
        elif len(nargs) == 1:
            x = nargs[0]
            return self.hyps[0] ** 2 * np.ones(x.shape[0])
        else:
            raise TypeError


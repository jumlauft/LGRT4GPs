import numpy as np
import warnings
import GPy
from src.btn import BTN
import copy

class LGRTN(BTN):

    def __init__(self, dx, dy, kernels=(), **kwargs):
        ''' Locally Growing Random Tree Node

        Args:
            div_method (string): Method to divide a dataset.
                                 Choices are 'median', 'mean', 'center'
            wo_ratio (float):    width/overlap ratio. must be > 1
            max_pts (int):       maximum number of points per leaf
            inf_method (string): Method to combine local predictions
                                 Choices 'moe' (Mixture of Experts)
            optimize_hyps (bool):Hyperparameter optimization
            lazy_training (bool):Wait with training until prediction is called

            kernels (tuple):      kernel function (one for each dy)
        '''
        super().__init__()
        self.dx, self.dy = dx, dy
        options = kwargs
        options.setdefault('wo_ratio', 100)
        options.setdefault('div_method', 'center')
        options.setdefault('inf_method', 'moe')
        options.setdefault('max_pts', 100)
        options.setdefault('optimize_hyps', False)
        options.setdefault('lazy_training', False)
        self.opt = options

        if len(kernels) == 0:
            self.kernels = tuple([GPy.kern.RBF(input_dim=dx) for _ in range(dy)])
        else:
            self.kernels = tuple(kernels)

        # Dimension along which a data set is split
        # Training data
        self.X = np.empty((0, self.dx))
        self.Y = np.empty((0, self.dy))

        # GP model
        self.gps = []
        self._update_gp = True

    # function to add a data point to the binary tree
    def add_data(self, x, y):
        if x.size > 0:
            if not self.is_leaf():
                # if node is not a leaf, distribute data to children
                self._distribute_data(x, y)
            elif self.X.shape[0] + x.shape[0] > self.opt['max_pts']:
                # if node is leaf, but full divide
                self._divide(x, y)
            else:  # if node is a leaf and not full, add data
                self.X = np.vstack((self.X, x))  # add to data set
                self.Y = np.vstack((self.Y, y))
                if self.opt['lazy_training']:
                    self._update_gp = True
                else:
                    self._setup_gps()
                    self._update_gp = False
                self.str = 'N=' + str(self.X.shape[0])

    def is_full(self):
        return self.X.shape[0] >= self.opt['max_pts']

    def _divide(self, x, y):
        # Total data to distribute
        X = np.vstack((self.X, x))
        Y = np.vstack((self.Y, y))

        # determine cutting dimension
        self.div_dim, self.div_val, self.ol = self._get_divider(X)

        # create empty child GPs
        self.set_leftChild(LGRTN(self.dx, self.dy, **self.opt,
                                 kernels=tuple(copy.deepcopy(self.kernels))))
        self.set_rightChild(LGRTN(self.dx, self.dy, **self.opt,
                                 kernels=tuple(copy.deepcopy(self.kernels))))

        # Pass data
        self._distribute_data(X, Y)

        # empty parent GP
        self.X, self.Y = np.empty((0, self.dx)), np.empty((0, self.dy))
        self.gps, self.kernels = [], []

    def _distribute_data(self, x, y):
        # compute probabilities and sample assignment
        ileft = np.random.binomial(1, self._prob_left(x)).astype(bool)
        iright = np.invert(ileft)

        # assign data to children
        self.get_leftChild().add_data(x[ileft, :], y[ileft, :])
        self.get_rightChild().add_data(x[iright, :], y[iright, :])

    def _prob_left(self, X):
        return np.clip(0.5 + (X[:, self.div_dim] - self.div_val) / self.ol, 0,
                       1)

    def _get_divider(self, X):
        """ Returns division dimension, value and overlap
        """
        ma, mi = X.max(axis=0), X.min(axis=0)
        width = ma - mi
        dim = np.argmax(width)

        if self.opt['div_method'] == 'median':
            val = np.median(X[:, dim])
        elif self.opt['div_method'] == 'mean':
            val = np.mean(X[:, dim])
        else:
            val = (ma[dim] + mi[dim]) / 2
            if self.opt['div_method'] != 'center':
                warnings.warn(
                    "Division method unknown, used 'center' by default")

        if width[dim] == 0:
            warnings.warn("Split along a dimension of width 0, set o = 0.1")
            o = 0.1
        else:
            o = (ma[dim] - mi[dim]) / self.opt['wo_ratio']
        self.str = 'x_' + str(dim) + '<' + '{:f}'.format(val)
        return dim, val, o

    def predict(self, xt):
        mus, s2s, eta = [], [], []
        self.predict_local(xt, mus, s2s, eta, np.zeros(xt.shape[0]))
        mus, s2s = np.stack(mus, axis=2), np.stack(s2s, axis=2)
        w = np.exp(np.stack(eta, axis=1).reshape(-1, 1, len(eta)))
        if self.opt['inf_method'] != 'moe':
            warnings.warn("inference method unknown, used 'moe' as default")

        mu = np.sum(mus * w, axis=2)
        s2 = np.sum(w * (s2s + mus ** 2), axis=2) - mu ** 2
        return mu, s2

    def predict_local(self, xt, mu, s2, eta, log_p):
        if self.is_leaf():
            m, v = self._compute_mus2(xt)
            mu.append(m)
            s2.append(v)
            eta.append(log_p)
        else:
            # compute marginal probabilities
            mpl = self._prob_left(xt)
            mpr = 1 - mpl

            # compute probabilities
            np.seterr(divide='ignore')
            log_pl, log_pr = np.log(mpl) + log_p, np.log(mpr) + log_p
            np.seterr(divide='warn')

            # if nonzero probabilities exist, pass on to children
            if np.any(log_pl > -np.inf):
                self.get_leftChild().predict_local(xt, mu, s2, eta, log_pl)
            if np.any(log_pr > -np.inf):
                self.get_rightChild().predict_local(xt, mu, s2, eta, log_pr)

    def _setup_gps(self):
        self.gps = []
        for dy in range(self.dy):
            gp = GPy.models.GPRegression(self.X, self.Y[:, dy:dy + 1],
                                         self.kernels[dy])
            if self.opt['optimize_hyps']:
                gp.optimize(messages=False)
            self.gps.append(gp)

    def _compute_mus2(self, xt):
        if self._update_gp:
            self._setup_gps()
        mus, s2s = [], []
        for dy in range(self.dy):
            mu, s2 = self.gps[dy].predict(xt)
            mus.append(mu), s2s.append(s2)
        return np.concatenate(mus, axis=1), np.concatenate(s2s, axis=1)

    def __len__(self):
        return self.num_leaves()

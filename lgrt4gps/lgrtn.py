import numpy as np
import warnings
from lgrt4gps.btn import BTN
import copy
import multiprocessing as mp


def wrapper(input):
    fun, args = input
    return fun(*args)

class LGRTN(BTN):
    """
    Locally Growing Random Tree Node inherits from BTN

    .. versionadded:: 0.0.1

    Attributes
    ----------
    dx : int
        input dimension
    dy : int
        output dimension
    kernels : list (length: dy)
        list of kernels from GPy.kern
    GP_engine : str, optional
        Default is '' for simple high-speed  GP or 'GPy' using the GPy package
    div_method : string, optional
        Method to divide a dataset.
        Choices are 'median', 'mean', 'center'
    wo_ratio : float, optional
        width/overlap ratio. must be > 1
    max_pts : int, optional
        maximum number of points per leaf
    inf_method : string, optional
        Method to combine local predictions
        Choices 'moe' (Mixture of Experts)
    optimize_hyps : bool
        Turn hyperparameter optimization on or off
    lazy_training : bool
        Wait with training until prediction is called
    multi_processing : bool
        enables parrallel processing (for 1e5 datapoints overhead is too big)
     """

    def __init__(self, dx, dy, kerns=(), GP_engine='', div_method='center',
                 wo_ratio=100, max_pts=100, inf_method='moe',
                 optimize_hyps=False, lazy_training=False,
                 multi_processing = False, **kwargs):
        """
        Locally Growing Random Tree Node

        """
        super().__init__(**kwargs)
        self.dx, self.dy = dx, dy
        self.opt = {
            'GP_engine': GP_engine,
            'div_method': div_method,
            'wo_ratio': wo_ratio,
            'max_pts' :max_pts ,
            'inf_method':inf_method,
            'optimize_hyps':optimize_hyps,
            'lazy_training': lazy_training,
            'multi_processing': multi_processing
        }

        # Load correct GP engine and corresponding kernel classes
        if self.opt['GP_engine'] == '':
            from lgrt4gps.gp import GP
            self._gp_class = GP
            from lgrt4gps.kern import RBF
            self._kernel_class = RBF
        elif self.opt['GP_engine'] == 'GPy':
            from GPy.models import GPRegression
            self._gp_class = GPRegression
            from GPy.kern import RBF
            self._kernel_class = RBF
        else:
            raise NotImplementedError

        # Generate default kernels or check provided kernels
        if len(kerns) == 0:
            self._kernels = tuple(
                [self._kernel_class(input_dim=dx) for _ in range(dy)])
        else:
            if self.dy != len(kerns):
                raise ValueError('incorrect number of kernels')
            self._kernels = copy.deepcopy(kerns)
            for k in self._kernels:
                if not isinstance(k, self._kernel_class):
                    raise TypeError
                if k.input_dim != self.dx:
                    raise ValueError

        # Training data
        self._X = np.empty((0, self.dx))
        self._Y = np.empty((0, self.dy))

        # GP model
        self._gps = []
        self._update_gp = True

    @property
    def gps(self):
        """
        list of GP instances (one for each output dimension), read only

        Returns
        -------
        gps : list
            list of GP instances
        """
        return self._gps

    @gps.setter
    def gps(self, gps):
        Exception("the GPs can only be changed at initialization")

    @property
    def kernels(self):
        """
        list of kernel instances (one for each output dimension), read only

        Returns
        -------
        kernels : list
        """
        return self._kernels

    @kernels.setter
    def kernels(self, kerns):
        Exception("the kernels can only be changed at initialization")

    @property
    def X(self):
        """
        input training data, read only

        Returns
        -------
        X : numpy array (ntr x dx)
            input training data
        """
        return self._X

    @X.setter
    def X(self, x):
        Exception("X may only be modified by add_data")

    @property
    def Y(self):
        """
        output training data, read only

        Returns
        -------
        Y : numpy array (ntr x dy)
            output training data
        """
        return self._Y

    @Y.setter
    def Y(self, y):
        Exception("Y may only be modified by add_data")
    def get_child_xy(self):
        """
        Recursively gathers training data from all leaves

        Returns
        -------
        childX : numpy array (ntr x dx)
            Input training data concatenated from all children
        childY : numpy array (ntr x dy)
            Output training data concatenated from all children
        """
        if self.is_leaf:
            return self.X, self.Y
        leftX, leftY = self.left.get_child_xy()
        rightX, rightY = self.right.get_child_xy()
        return np.vstack((leftX,rightX)), np.vstack((leftY,rightY))

    def add_data(self, x, y):
        """
            Adds data points to the node (or its children)

        Three options exist:
        1. node is not a leaf: distribute data to children
        2. node is a leaf, but full: grow two children and distribute
        3. node is a leaf is not full: add data to node

        Parameters
        ----------
        x : numpy array (n x dx)
            input data to be added
        y : numpy array (n x dx)
            output data to be added
        """
        if x.ndim != 2 or y.ndim != 2 or x.shape[1] != self.dx or \
                y.shape[1] != self.dy or x.shape[0] != y.shape[0]:
            raise ValueError('incorrect input dimensions')

        self.value += x.shape[0]
        if x.size > 0:
            if not self.is_leaf:
                # if node is not a leaf, distribute data to children
                self._distribute_data(x, y)
            elif self.X.shape[0] + x.shape[0] > self.opt['max_pts']:
                # if node is leaf, but full divide
                self._divide(x, y)
            else:  # if node is a leaf and not full, add data
                self._X = np.vstack((self.X, x))  # add to data set
                self._Y = np.vstack((self.Y, y))
                if self.opt['lazy_training']:
                    self._update_gp = True
                else:
                    self._setup_gps()
                self.str = 'N=' + str(self.X.shape[0])

    @property
    def is_full(self):
        """
        True if current node reached maximum capacity

        Returns
        -------
        is_full : bool
        """
        return self.X.shape[0] >= self.opt['max_pts']

    def fit(self):
        """
        Prepares the model for prediction

        For lazy training it enforces generation of GPs and trains the
        hyperparameters if needed

        """
        if self.is_leaf:
            if self._update_gp == True:
                self._setup_gps()
            if self.opt['optimize_hyps']:
                for gp in self.gps:
                    gp.optimize()
        else:
            self.left.fit()
            self.right.fit()

    def _divide(self, x, y):
        """
        The current node grows two children and distributes its data

        The node grows two children and distributes the new data (x,y) and its
        own data to the children

        Parameters
        ----------
        x : numpy array (n x dx)
        y : numpy array (n x dy)

        """
        # Total data to distribute
        X = np.vstack((self.X, x))
        Y = np.vstack((self.Y, y))

        # determine cutting dimension
        self.div_dim, self.div_val, self.ol = self._get_divider(X)

        # create empty child GPs
        self.left = LGRTN(self.dx, self.dy, kerns=self.kernels,
                        parent=self, **self.opt)
        # kernels=self.kernels)
        self.right = LGRTN(self.dx, self.dy, kerns=self.kernels,
                           parent=self, **self.opt)
        # kernels=self.kernels)

        # Pass data
        self._distribute_data(X, Y)

        # empty parent GP
        self._X, self._Y = np.empty((0, self.dx)), np.empty((0, self.dy))
        self._gps, self._kernels = [], []

    def _distribute_data(self, x, y):
        """
        Distributes data to children

        Parameters
        ----------
        x : numpy array (n x dx)
        y : numpy array (n x dy)
        """
        # compute probabilities and sample assignment
        ileft = np.random.binomial(1, self._prob_left(x)).astype(bool)
        iright = np.invert(ileft)

        # assign data to children
        self.left.add_data(x[ileft, :], y[ileft, :])
        self.right.add_data(x[iright, :], y[iright, :])

    def _prob_left(self, x):
        """
        Computes probabilities of for left child

        Parameters
        ----------
        x : numpy array (n x dx)

        Returns
        -------
        prob_left: numpy array (n,)
            probabilities for left child
        """
        return np.clip(0.5 + (self.div_val - x[:, self.div_dim]) / self.ol,
                        0, 1)

    def _get_divider(self, x):
        """
        Computes parameters required for division of data set

        1. Computes dimension with widest spread
        2. Computes division values 'median', 'mean', or 'center'
        3. Computes overlap region
        Parameters
        ----------
        x : numpy array (n x dx)

        Returns
        -------
        dim : int (1...dx)
            dividing dimension
        val : float
            dividing value
        o : float
            overlap
        """
        # Compute dimension with widest spread
        ma, mi = x.max(axis=0), x.min(axis=0)
        width = ma - mi
        dim = np.argmax(width)

        # Compute division value
        if self.opt['div_method'] == 'median':
            val = np.median(x[:, dim])
        elif self.opt['div_method'] == 'mean':
            val = np.mean(x[:, dim])
        elif self.opt['div_method'] == 'center':
            val = (ma[dim] + mi[dim]) / 2
        else:
            raise NotImplemented

        if width[dim] == 0:
            warnings.warn("Split along a dimension of width 0, set o = 0.1")
            o = 0.1
        else:
            o = (ma[dim] - mi[dim]) / self.opt['wo_ratio']
        self.str = 'x_' + str(dim) + '<' + '{:f}'.format(val)
        return dim, val, o

    def predict(self, xt):
        """
        Predicts posterior mean and variance based on all leaf GPs

        Starts the recursive call to  prediction functions of leaves
        and combines the predictions according to the chosen inference method

        Parameters
        ----------
        xt : numpy array (n x dx)
            test inputs

        Returns
        -------
        mu : numpy array (n x dy)
            posterior mean
        s2 : numpy array (n x dy)
            posterior variance
        """
        if xt.ndim != 2 or xt.shape[1] != self.dx:
            raise ValueError('Incorrect input dimension')

        # Recursively get predictions from all leaves
        mus, s2s, etas = [], [], []
        nte = xt.shape[0]

        mus, s2s, etas = self.predict_local(xt, np.zeros(nte))

        # Combine predicted values based on inference method
        mu, s2 = np.zeros((nte, self.dy)), np.zeros((nte, self.dy))

        if self.opt['inf_method'] == 'moe':
            for (m, s, e) in zip(mus, s2s, etas):
                idx = np.invert(np.isinf(e))
                w = np.exp(e[idx].reshape(-1, 1))
                mu[idx, :] += m * w
                s2[idx, :] += w * (s + m ** 2)
            s2 -= mu ** 2
        elif self.opt['inf_method'] == 'simple':
            for (m, s, e) in zip(mus, s2s, etas):
                idx = np.invert(np.isinf(e))
                w = np.exp(e[idx].reshape(-1, 1))
                mu[idx, :] += m * w
                s2[idx, :] += w * s
        else:
            raise NotImplementedError
        return mu, s2

    def predict_local(self, xt, log_p):
        """
        Local prediction function

        If current node is leaf, compute posterior estimates. If is not leaf,
        recurse further if any child has non-zero probability

        Parameters
        ----------
        xt : numpy array (n x dx)
            input test points
        mu : list
            list of posterior mean predictions of leaves
        s2 : list
            list of posterior variance predictions of leaves
        eta : list
            list of log probabilities to reach the leaf
        log_p :
            log probability to reach the leaf

        """
        mu, s2, eta = [], [], []
        if self.is_leaf:
            idx = np.invert(np.isinf(log_p))
            m, v = self._compute_mus2(xt[idx, :])
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
            if mp.cpu_count() > 1 \
                    and self.is_root and self.opt['multi_processing']:
                tasks = []
                if np.any(log_pl > -np.inf):
                    tasks.append((self.left.predict_local,(xt, log_pl)))
                if np.any(log_pr > -np.inf):
                    tasks.append((self.right.predict_local,(xt, log_pr)))
                if len(tasks) > 0:
                    with mp.Pool(2) as pool:
                        res = pool.map(wrapper, tasks, 1)
                    for r in res:
                        mu.extend(r[0]), s2.extend(r[1]), eta.extend(r[2])
            else:
                if np.any(log_pl > -np.inf):
                    lmu, ls2, leta = self.left.predict_local(xt, log_pl)
                    mu.extend(lmu), s2.extend(ls2), eta.extend(leta)
                if np.any(log_pr > -np.inf):
                    rmu, rs2, reta = self.right.predict_local(xt, log_pr)
                    mu.extend(rmu), s2.extend(rs2), eta.extend(reta)

        return mu, s2, eta


    def _setup_gps(self):
        """
            Prepare gp models (one for each output)
        """
        self._gps = []
        for dy in range(self.dy):
            gp = self._gp_class(self.X, self.Y[:, dy:dy + 1], self.kernels[dy])
            if self.opt['optimize_hyps']:
                gp.optimize(messages=False)
            self._gps.append(gp)
        self._update_gp = False

    def _compute_mus2(self, xt):
        """
        Calls gp from GPy to make predictions

        Parameters
        ----------
        xt : numpy array (n x dx)
            test inputs

        Returns
        -------
        mu : numpy array (n x dy)
            posterior mean
        s2 : numpy array (n x dy)
            posterior variance
        """
        if self._update_gp:
            self._setup_gps()
        mus, s2s = [], []
        for dy in range(self.dy):
            mu, s2 = self.gps[dy].predict(xt)
            mus.append(mu), s2s.append(s2)
        return np.concatenate(mus, axis=1), np.concatenate(s2s, axis=1)

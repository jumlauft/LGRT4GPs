Overview
--------

A minimal working example

.. code-block:: python
    
    >>> import numpy as np
    >>> from lgrt4gps.lgrtn import LGRTN
    
    >>> dx, dy = 1, 1
    >>> ntr, nte = 500, 100
    >>> f = lambda x: np.sin(x[:, :1])
    
    >>> xtr = np.random.uniform(0,1,size=(ntr, dx))
    >>> xte = np.random.uniform(0,1,size=(nte, dx))
    >>> ytr, yte = f(xtr), f(xte)
    
    >>> lgrt_gp = LGRTN(dx, dy)
    >>> lgrt_gp.add_data(xtr,ytr)
    >>> mu_lgrt_gp, s2_lgrt_gp = lgrt_gp.predict(xte)
    >>> mse = np.mean((mu_lgrt_gp - yte) ** 2)
    >>> print(lgrt_gp)
    
                      _____________________________x_0<0.500304____________________________
                 /                                                                     \
         __x_0<0.749385__________                                       ___________x_0<0.252210__________
        /                        \                                     /                                 \
      N=95                 __x_0<0.623854_                     __x_0<0.378240_                     __x_0<0.125555_
                          /               \                   /               \                   /               \
                        N=62              N=78              N=67              N=64              N=61              N=73
      
      >>> print(mse)
      0.0008375186565166093
    

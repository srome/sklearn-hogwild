import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.externals.joblib import Parallel, delayed

from shared import SharedWeights, mse_gradient_step
from generators import DataGenerator

class HogWildRegressor(SGDRegressor):
    """
    Class to implement a variant of Hogwild! from
    "Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent".

    Class follows the sklearn interface. Use get_SGDRegressor() to transfer
    the weights to a SDGRegressor for pickling.

    Parameters: 
        batch_size : Number of examples to compute the gradient
        chunk_size : Minibatch size sent to each workers
        learning_rate : Learning rate used in SGDRegressor
        n_epochs : Number of times to loop over the data
        n_jobs : Number of parallel workers
        generator : None will default to the DataGenerator class

    Supported sklearn Parameters:
        loss
        verbose
        shuffle


    Recommendations:
    - batch_size = 1 / chunk_size = 1 is the same as the original paper. 
    - batch_size = 1 / chunk_size ~ small (i.e. 32) seems to enhance performance.
    """

    losses = {
        'squared_loss' : mse_gradient_step
    }

    def __init__(self, 
                 n_jobs=-1, 
                 n_epochs = 5,
                 batch_size = 1, 
                 chunk_size = 32,
                 learning_rate = .001,
                 generator=None,
                 **kwargs):
        super(HogWildRegressor, self).__init__(**kwargs)

        if self.loss not in self.losses:
            raise Exception("Loss '%s' not supported")

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.gradient = self.losses.get(self.loss)
        self.n_jobs = n_jobs
        self.n_epochs = n_epochs
        self.chunk_size = chunk_size
        self.shared_weights = SharedWeights

        if not generator:
            self.generator = DataGenerator(shuffle= self.shuffle,
                                           chunk_size = self.chunk_size,
                                           verbose = self.verbose)

    def _fit(self, X, y, **kwargs):
        # Check y
        np.random.seed(self.random_state)
        y = y.reshape((len(y),1)) # proper shape for numpy descent
        size_w = X.shape[1]

        # Create module to properly share variables between processes
        with self.shared_weights(size_w=X.shape[1]) as sw:
            for epoch in range(self.n_epochs):
                if self.verbose:
                    print('Epoch: %s' % epoch)
                Parallel(n_jobs= self.n_jobs, verbose=self.verbose)\
                            (delayed(self.train_epoch)(e) for e in self.generator(X,y))

        self.coef_ = sw.w.reshape((10,1)).T
        self.fitted = True
        self.intercept_ = 0.
        self.t_ = 0.

        return self

    def train_epoch(self, inputs):
        X,y = inputs
        self._train_epoch(X,y)

    def _train_epoch(self, X, y):
        batch_size = self.batch_size
        for k in range(int(X.shape[0]/float(batch_size))):
            Xx = X[k*batch_size : (k+1)*batch_size,:]
            yy = y[k*batch_size : (k+1)*batch_size]
            self.gradient(Xx,yy,self.learning_rate)


    def get_SGDRegressor(self):
        sr = SGDRegressor(fit_intercept = False)
        sr.coef_ = self.coef_
        sr.intercept_ = 0.
        self.t_ = 0
        return sr





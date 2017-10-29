import unittest
from hogwildsgd import HogWildRegressor
import scipy.sparse
import numpy as np


class TestHogwild(unittest.TestCase):

    def test_work(self):
        X = scipy.sparse.random(20000,10, density=.2).toarray() # Guarantees sparse grad updates
        real_w = np.random.uniform(0,1,size=(10,1))
        y = np.dot(X,real_w)


        hw = HogWildRegressor(n_jobs = 4, 
                              n_epochs = 5,
                              batch_size = 1, 
                              chunk_size = 32,
                              learning_rate = .001,
                              generator=None,
                              verbose=2)
        hw = hw.fit(X,y)


        y_hat = hw.predict(X)
        y = y.reshape((len(y),))
        score = np.mean(abs(y-y_hat))
        self.assertTrue(score < .005) 


if __name__ == '__main__':
    unittest.main()
import numpy as np

"""
This file defines the Model class, which defines utilities for evaluating error
rate and generating learning curves, and class for Bayesian logistic regression
"""

class Model(object):
    """
    Parent class for models. Defines evaluation utilities
    """
    
    def test(self, X, y, add_dim=False):
        """
        Evaluate error rate of trained model
        Parameters:
            array X: (N,M) data with N instances and M features
            array y: (N,) labels
            bool add_dim: concatenate column of 1's to fit intercept if true
        """
        if add_dim:
            X = X.copy()
            X = np.hstack((np.ones(len(X))[:,None], X))
        pp = self.predict(X, add_dim=False).ravel()
        pp_pred = (pp >=.5).astype(int)
        
        return (pp_pred != y).astype(int).sum() / len(y)
    
    
    def learning_curve(self, X, y, args={}, other_params=[], runs=30, m=10, test_size=.4,
                       add_dim=False, max_train_size=np.inf):
        """
        Generate learning curve for model
        Parameters:
            array X: (N,M) data with N instances and M features
            array y: (N,) labels
            dict args: parameters to pass to models fit function
            list other_params: other parameters to keep track of
            int runs: number of runs to generate average error
            int m: number of training sizes to test
            bool add_dim: concatenate column of 1's to fit intercept if true
        """
        
        n = X.shape[0]
        test_size = int(n*test_size)
        train_size = min(n - test_size, max_train_size)
        min_size = int(n/m)
        step = (train_size - min_size)/(m-1)
        train_sizes = np.arange(min_size, train_size+step, step).astype(int)
        all_error_rates = []
        all_params = {k:[] for k in other_params}
        i = 0
        while i < runs:
            try:
                random_idx = np.arange(len(X))
                np.random.shuffle(random_idx)
                shuffled_X = X[random_idx,:]
                shuffled_y = y[random_idx]
                
                testX = shuffled_X[:test_size]
                testy = shuffled_y[:test_size]
                
                error_rates = []
                params = {k:[] for k in other_params}
                for j in train_sizes:
                    trainX = shuffled_X[test_size: test_size+j]
                    trainy = shuffled_y[test_size: test_size+j]
                    
                    self.fit(trainX, trainy, **args)
                    error_rate = self.test(testX, testy, add_dim=add_dim)
                    error_rates.append(error_rate)
                    for param in other_params:
                        params[param].append(getattr(self, param))
                    
                all_error_rates.append(error_rates)
                for param in other_params:
                        all_params[param].append(params[param])
            #restart run if ipdb import error from GPy
            except Exception as e:
                if e.msg == "No module named 'ipdb'":
                    continue
                else:
                    raise e
            i+=1
        
        all_error_rates = np.array(all_error_rates)
        self.learning_curve_mean = all_error_rates.mean(axis=0)
        self.learning_curve_std = all_error_rates.std(axis=0)
        self.learning_curve_sizes = train_sizes
        self.all_params_mean = {k: np.array(all_params[k]).mean(axis=0) for k in all_params.keys()}
        self.all_params_std= {k: np.array(all_params[k]).std(axis=0) for k in all_params.keys()}
    

class LogisticRegression(Model):
    """
    Bayesian logistic regression
    """
    def prob(self, X, w, add_dim=False):
        """
        Get predictive probability for X given w
            array X: (N,M) data with N instances and M features
            array w: (M,) weights
            bool add_dim: concatenate column of 1's to fit intercept if true
        """
        if add_dim:
            X = X.copy()
            X = np.hstack((np.ones(len(X))[:,None], X))
        z = np.dot(X, w)
        p = np.exp(z)
        return p / (1. + p)
    
    
    def grad_log_posterior(self, X, y, w, alpha):
        """
        Return gradient of the log map likelihood
            array X: (N,M) data with N instances and M features
            array y: (N,) labels
            array w: (M,) weights
            float alpha: prior (precision) on weights
        """
        mu = self.prob(X, w)
        H = np.eye(X.shape[1])*alpha
        grad = np.dot(X.T, (mu - y)) + np.dot(H, w)
        return grad
    
    
    def hess_log_posterior(self, X, y, w, alpha):
        """
        Return Hessian of the log map likelihood
            array X: (N,M) data with N instances and M features
            array y: (N,) labels
            array w: (M,) weights
            float alpha: prior (precision) on weights
        """
        mu = self.prob(X, w)
        S = np.eye(X.shape[1])*alpha
        R = mu * (1. - mu)
        hess = np.dot(X.T, X * R[:,None]) + S
        return hess
    
    
    def newton_raphson(self, X, y, alpha, maxiter=100):
        """
        Perform newton-raphson optimization to find the optimal value of w
            array X: (N,M) data with N instances and M features
            array y: (N,) labels
            float alpha: prior (precision) on weights
            int maxiter: maximum number of iterations
        """
        w = np.zeros(X.shape[1])
        for i in range(maxiter):
            grad = self.grad_log_posterior(X, y, w, alpha)
            hess = self.hess_log_posterior(X, y, w, alpha)
            inv_hess = np.linalg.inv(hess)
            delta = np.dot(inv_hess, grad)
            new_w = w - delta
            eps = ((new_w - w)**2).sum() / (new_w**2).sum()
            if eps < 10**-3:
                w = new_w
                break
            w = new_w
        #print ('Newton-Raphson converged in ' + str(i) + ' iterations')
        return w
    
    
    def fit(self, X, y, alpha=1, maxiter=100, maximize_evidence=False):
        """
        Fit logistic regression model
        Parameters:
            array X: (N,M) data with N instances and M features
            array y: (N,) labels
            float alpha: prior (precision) on weights
            int maxiter: maximum number of iterations
            bool maximize_evidence: find optimal value of alpha if true
        """
        if maximize_evidence:
            self.maximize_evidence(X, y)
        else:
            X = X.copy()
            X = np.hstack((np.ones(len(X))[:,None], X))
            w = self.newton_raphson(X, y, alpha, maxiter)
            S = self.hess_log_posterior(X, y, w, alpha)
            self.w = w
            self.S = S
        
        
    def predict(self, X, add_dim=True):
        """
        Get p(y=1|X) for new X
        Parameters:
            array X: (N,M) data with N instances and M features
            array y: (N,) labels
            bool add_dim: concatenate column of 1's to fit intercept if true
        """
        if add_dim:
            X = X.copy()
            X = np.hstack((np.ones(len(X))[:,None], X))
        S_inv = np.linalg.inv(self.S)
        sig = np.sum(X * np.dot(S_inv, X.T).T, axis=1)
        noise = 1. / np.sqrt(1. + 0.125 * np.pi * sig)
        z = np.dot(X, self.w)
        post_z = z * noise
        pr = np.exp(post_z)
        return pr / (1 + pr)
    
    
    def maximize_evidence(self, X, y):
        """
        Find optimal prior alpha
        Parameters:
            array X: (N,M) data with N instances and M features
            array y: (N,) labels
        """
        alpha = 1
        for l in range(10):
            self.fit(X, y, alpha)
            
            mu = self.prob(X, self.w, add_dim=True)
            S = mu * (1. - mu)
            H = np.dot(X.T, X * S[:, np.newaxis])
            eigenvals = np.linalg.eigvals(H).astype(float)
            gamma = (eigenvals / (eigenvals + alpha)).sum()
            alpha = gamma / (mu**2).sum()
            
        self.fit(X, y, alpha)
        self.alpha = alpha
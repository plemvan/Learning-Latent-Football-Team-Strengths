#========== Classical Bradley-Terry Model ==========#

## Imports
import numpy as np
from functools import partial
from Gradient_descent import GradientDescent


#===================================================#

class BradleyTerry():

    """Class implementing the classical Bradley-Terry framework"""

    def __init__(self,learning_rate : float = 0.01, n_iterations : int = 1000, tolerance : float = 1e-6):
        
        """
        Bradley-Terry model

        Parameters
        ----------
        learning_rate : float. Default = 0.01
            Step of the gradient descent
        n_iterations : int. Default = 1000
            Number of iterations of the algorithm
        tolerance : float. Default = 1e-6
            Stop criterion (if the change is less than this threshold)
        
        """

        # Hyperparameters for the gradient descent
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.tolerance = tolerance

        return
    
    def loglikelihood(self,X : np.array, theta : np.array)-> float:

        """
        Compute and return the log-likelihood of the model for a matrix of results X and a vector of strengths theta

        Parameters
        ----------
        X : np.array
            Matrix of results, for i<j, Xij = 1 if i wins, 0 if j wins. Whatever for other i,j
        theta : np.array
            Vector of strengths

        Returns
        -------
        loglik : float.
            Value of the log-likelihood for the given X and theta
        """

        X = np.asarray(X)
        theta = np.asarray(theta)

        n = len(theta)

        # Building of theta_i - theta_j for all i<j
        theta_diff = theta[:,None] - theta[None,:]

        # Mask for the upper triangular matrix i<j
        mask = np.triu(np.ones((n,n), dtype=bool), k=1)

        td = theta_diff[mask]
        Xij = X[mask]

        loglik = np.sum(Xij * td - np.log1p(np.exp(td)))

        return loglik


    def log_gradient(self,X: np.array, theta: np.array)-> np.array:

        """
        Compute and return the gradient of the log-likelihood of the model for a matrix of results X and a vector of strenghts theta

        Parameters
        ----------
        X : np.array
            Matrix of results, for i<j, Xij = 1 if i wins, 0 if j wins. Whatever for other i,j
        theta : np.array
            Vector of strengths
        
        Returns
        -------
        grad : np.array
            Value of the gradient of the log-likelihood for the given X and theta
        """

        X = np.asarray(X)
        theta = np.asarray(theta)

        n = len(theta)

        # Building of theta_i - theta_j for all i<j
        theta_diff = theta[:,None] - theta[None,:]

        # Building p_ij = sigmoid(theta_i - theta_j)
        p = 1 / (1+ np.exp(-theta_diff))

        # Mask for the upper triangular matrix i<j
        mask = np.triu(np.ones((n,n), dtype=bool), k=1)

        Xij = X[mask]
        pij = p[mask]

        grad = np.zeros(n)
        
        # Splitting contribution given the position of the index
        i_idx, j_idx = np.where(mask)

        grad_i_contrib = Xij - pij
        grad_j_contrib = -grad_i_contrib

        np.add.at(grad, i_idx, grad_i_contrib)
        np.add.at(grad, j_idx, grad_j_contrib)

        return grad


    def add_gradient(self,X: np.array):

        """
        Add the function theta -> log_gradient(X, theta) to the attributes of the class

        Parameters
        ----------
        X : np.array
            Matrix of results, for i<j, Xij = 1 if i wins, 0 if j wins. Whatever for other i,j
        
        Returns
        -------
        Add an attribute to the instance of BradleyTerry
        """

        self.gradient: callable = partial(self.log_gradient, X=X)

        return


    def fit(self,X : np.array) -> BradleyTerry:

        """
        Fit the model to the data

        Parameters
        ----------
        X : np.array
            Matrix of results, for i<j, Xij = 1 if i wins, 0 if j wins. Whatever for other i,j
        Returns
        -------
        self : object
        """

        self.theta = np.zeros(X.shape[0])

        self.add_gradient(X=X)

        # Gradient descent
        Optimizer = GradientDescent(learning_rate=self.learning_rate,
                                    n_iterations=self.n_iterations,
                                    tolerance=self.tolerance)
        
        self.theta = Optimizer.optimize(gradient=self.gradient,
                                        starting_point= self.theta)

        return self
    
    def predict(self) -> np.array:

        """
        Return the strengths of each team

        Returns
        -------
        theta : np.array
            Array of strengths
        """

        return self.theta


#========== Classical Bradley-Terry Model ==========#

## Imports
import numpy as np
from functools import partial
from scipy.special import expit
from .Gradient_descent import GradientDescent


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

        # Attributes
        self.X : np.array = None # Matrix of results
        self.theta : np.array = None # Vector of strengths

        self.teams = None
        self.teams_index : dict = None

        return
    

    def loglikelihood(self, theta : np.array)-> float:

        """
        Compute and return the log-likelihood of the model for a vector of strengths theta and the matrix of results X

        Parameters
        ----------
        theta : np.array
            Vector of strengths

        Returns
        -------
        loglik : float
            Value of the log-likelihood for the given X and theta
        """

        X = self.X
        theta = np.asarray(theta)

        n = len(theta)

        loglik = 0.0

        for i in range(n):
            for j in range(i+1, n):
                diff = theta[i] - theta[j]
                loglik += X[i,j]*diff
                loglik -= (X[i,j] + X[j,i])*np.log1p(np.exp(diff))

        return loglik


    def log_gradient(self, theta: np.array)-> np.array:

        """
        Compute and return the -(gradient of the log-likelihood) of the model for a vector of strenghts theta and the matrix of results X
        (We compute -gradient instead of +gradient to conduct a gradient descent and not a gradient ascent)

        Parameters
        ----------
        theta : np.array
            Vector of strengths
        
        Returns
        -------
        grad : np.array
            Value of the gradient of the log-likelihood for the given X and theta
        """

        X = self.X
        theta = np.asarray(theta)

        n = len(theta)

        grad = np.zeros(n, dtype=float)

        for k in range(n):
            for j in range(n):
                if j==k:
                    continue

                diff = theta[k] - theta[j]

                # Computing sigmoïd via scipy.special.expit (more stability)
                p = expit(diff)

                Nkj = X[k,j] + X[j,k]

                grad[k] -= X[k,j] - Nkj*p # Substract to get -gradient 

        return grad


    def fit(self, X : np.array, teams : list) -> BradleyTerry:

        """
        Fit the model to the data

        Parameters
        ----------
        X : np.array
            Matrix of results, for i != j, X[i,j] is the number of victory of i against j

        Returns
        -------
        self : object
        """

        self.theta = np.zeros(X.shape[0])
        self.X = X

        self.teams = teams
        self.teams_index = {team:i for i, team in enumerate(teams)}


        # Gradient descent
        Optimizer = GradientDescent(learning_rate=self.learning_rate,
                                    n_iterations=self.n_iterations,
                                    tolerance=self.tolerance)
        
        self.theta = Optimizer.optimize(gradient=self.log_gradient,
                                        starting_point= self.theta)
        
        self.theta -= np.mean(self.theta)

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

